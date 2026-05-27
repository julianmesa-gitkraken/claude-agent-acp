import { describe, it, expect, vi } from "vitest";
import type {
  SDKMessage,
  SDKResultMessage,
  SDKTaskNotificationMessage,
  SDKTaskStartedMessage,
} from "@anthropic-ai/claude-agent-sdk";
import { TurnQueue, OffTurnFollowupCollector, classifyOffTurn } from "../session-reader.js";

function silentLogger() {
  return { log: vi.fn(), error: vi.fn() };
}

// Build the smallest SDKMessage shape vitest will accept. We cast to the
// real type via `as unknown as SDKMessage` so call sites stay strongly
// typed.
function asMessage<T extends Record<string, unknown>>(m: T): SDKMessage {
  return m as unknown as SDKMessage;
}

describe("TurnQueue", () => {
  it("returns a message pushed before take()", async () => {
    const q = new TurnQueue();
    q.push(asMessage({ type: "assistant", uuid: "1", session_id: "s" }));
    const r = await q.take();
    expect(r.done).toBe(false);
    expect((r.value as { type: string }).type).toBe("assistant");
  });

  it("resolves the consumer when push happens after take()", async () => {
    const q = new TurnQueue();
    const p = q.take();
    q.push(asMessage({ type: "user", uuid: "2", session_id: "s" }));
    const r = await p;
    expect(r.done).toBe(false);
    expect((r.value as { type: string }).type).toBe("user");
  });

  it("preserves FIFO order", async () => {
    const q = new TurnQueue();
    q.push(asMessage({ type: "a", n: 1 }));
    q.push(asMessage({ type: "b", n: 2 }));
    q.push(asMessage({ type: "c", n: 3 }));
    expect(((await q.take()).value as { n: number }).n).toBe(1);
    expect(((await q.take()).value as { n: number }).n).toBe(2);
    expect(((await q.take()).value as { n: number }).n).toBe(3);
  });

  it("returns done:true on take() after close()", async () => {
    const q = new TurnQueue();
    q.close();
    const r = await q.take();
    expect(r.done).toBe(true);
  });

  it("resolves a pending take() with done:true when close() arrives", async () => {
    const q = new TurnQueue();
    const p = q.take();
    q.close();
    const r = await p;
    expect(r.done).toBe(true);
  });

  it("rejects a pending take() when error() arrives", async () => {
    const q = new TurnQueue();
    const p = q.take();
    q.error(new Error("boom"));
    await expect(p).rejects.toThrow("boom");
  });

  it("rejects subsequent take() after error() with no pending waiter", async () => {
    const q = new TurnQueue();
    q.error(new Error("nope"));
    await expect(q.take()).rejects.toThrow("nope");
  });

  it("throws on concurrent take()", () => {
    const q = new TurnQueue();
    void q.take();
    expect(() => q.take()).toThrow(/concurrent take/);
  });

  it("drops messages pushed after close() instead of throwing", async () => {
    const q = new TurnQueue();
    q.close();
    q.push(asMessage({ type: "a" }));
    const r = await q.take();
    expect(r.done).toBe(true);
  });

  it("close() and error() are idempotent", async () => {
    const q1 = new TurnQueue();
    q1.close();
    q1.close();
    expect((await q1.take()).done).toBe(true);

    const q2 = new TurnQueue();
    q2.error(new Error("first"));
    q2.error(new Error("second"));
    await expect(q2.take()).rejects.toThrow("first");
  });

  it("delivers buffered messages before reporting close() to consumer", async () => {
    const q = new TurnQueue();
    q.push(asMessage({ type: "a", n: 1 }));
    q.push(asMessage({ type: "b", n: 2 }));
    q.close();
    expect(((await q.take()).value as { n: number }).n).toBe(1);
    expect(((await q.take()).value as { n: number }).n).toBe(2);
    expect((await q.take()).done).toBe(true);
  });

  it("clear() drops buffered messages without closing the queue", async () => {
    const q = new TurnQueue();
    q.push(asMessage({ type: "a" }));
    q.push(asMessage({ type: "b" }));
    q.clear();
    const p = q.take();
    q.push(asMessage({ type: "c" }));
    const r = await p;
    expect(r.done).toBe(false);
    expect((r.value as { type: string }).type).toBe("c");
  });
});

describe("classifyOffTurn", () => {
  it("classifies task_started as lifecycle", () => {
    const m = asMessage<Partial<SDKTaskStartedMessage>>({
      type: "system",
      subtype: "task_started",
      task_id: "t1",
      description: "do thing",
    });
    expect(classifyOffTurn(m)).toBe("lifecycle");
  });

  it("classifies task_notification as lifecycle", () => {
    const m = asMessage<Partial<SDKTaskNotificationMessage>>({
      type: "system",
      subtype: "task_notification",
      task_id: "t1",
      status: "completed",
      summary: "ok",
      output_file: "/tmp/x",
    });
    expect(classifyOffTurn(m)).toBe("lifecycle");
  });

  it("classifies session_state_changed:idle as followup-candidate (collector handles it)", () => {
    const m = asMessage({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
    });
    expect(classifyOffTurn(m)).toBe("followup-candidate");
  });

  it("classifies assistant as followup-candidate", () => {
    expect(classifyOffTurn(asMessage({ type: "assistant" }))).toBe("followup-candidate");
  });

  it("classifies result as followup-candidate", () => {
    expect(classifyOffTurn(asMessage({ type: "result" }))).toBe("followup-candidate");
  });

  it("classifies stream_event as followup-candidate", () => {
    expect(classifyOffTurn(asMessage({ type: "stream_event" }))).toBe("followup-candidate");
  });

  it("classifies other system subtypes as followup-candidate", () => {
    expect(classifyOffTurn(asMessage({ type: "system", subtype: "hook_progress" }))).toBe(
      "followup-candidate",
    );
  });
});

describe("OffTurnFollowupCollector", () => {
  function setup(emitFollowup?: ReturnType<typeof vi.fn>) {
    const logger = silentLogger();
    const emit = emitFollowup ?? vi.fn().mockResolvedValue(undefined);
    const c = new OffTurnFollowupCollector(
      "s1",
      emit as unknown as (msgs: SDKMessage[], result: SDKResultMessage) => Promise<void>,
      logger,
    );
    return { c, emit, logger };
  }

  it("starts in idle state with empty buffer", () => {
    const { c } = setup();
    expect(c.inspect()).toEqual({ state: "idle", bufferSize: 0 });
  });

  it("accumulates non-terminal messages and moves to collecting state", async () => {
    const { c } = setup();
    await c.accept(asMessage({ type: "assistant", n: 1 }));
    expect(c.inspect()).toEqual({ state: "collecting", bufferSize: 1 });
    await c.accept(asMessage({ type: "stream_event", n: 2 }));
    expect(c.inspect()).toEqual({ state: "collecting", bufferSize: 2 });
  });

  it("emits followup when result has origin.kind=task-notification", async () => {
    const { c, emit } = setup();
    await c.accept(asMessage({ type: "assistant", n: 1 }));
    await c.accept(asMessage({ type: "stream_event", n: 2 }));
    const result = asMessage<Partial<SDKResultMessage>>({
      type: "result",
      origin: { kind: "task-notification" },
    });
    await c.accept(result);
    expect(emit).toHaveBeenCalledTimes(1);
    const [bufferedArg, resultArg] = emit.mock.calls[0];
    expect((bufferedArg as SDKMessage[]).length).toBe(2);
    expect(resultArg).toBe(result);
    expect(c.inspect()).toEqual({ state: "idle", bufferSize: 0 });
  });

  it("discards buffer with log when result has no task-notification origin", async () => {
    const { c, emit, logger } = setup();
    await c.accept(asMessage({ type: "assistant", n: 1 }));
    await c.accept(asMessage({ type: "result" }));
    expect(emit).not.toHaveBeenCalled();
    expect(logger.log).toHaveBeenCalledWith(
      expect.stringContaining("discarding 1 off-turn messages followed by non-followup result"),
    );
    expect(c.inspect()).toEqual({ state: "idle", bufferSize: 0 });
  });

  it("discards buffer with log on session_state_changed:idle without closing result", async () => {
    const { c, emit, logger } = setup();
    await c.accept(asMessage({ type: "assistant", n: 1 }));
    await c.accept(asMessage({ type: "stream_event", n: 2 }));
    await c.accept(
      asMessage({
        type: "system",
        subtype: "session_state_changed",
        state: "idle",
      }),
    );
    expect(emit).not.toHaveBeenCalled();
    expect(logger.log).toHaveBeenCalledWith(
      expect.stringContaining("discarding 2 off-turn messages with no closing result"),
    );
    expect(c.inspect()).toEqual({ state: "idle", bufferSize: 0 });
  });

  it("idle with empty buffer is a silent no-op", async () => {
    const { c, logger } = setup();
    await c.accept(
      asMessage({
        type: "system",
        subtype: "session_state_changed",
        state: "idle",
      }),
    );
    expect(logger.log).not.toHaveBeenCalled();
    expect(logger.error).not.toHaveBeenCalled();
  });

  it("ignores non-idle session_state_changed (treats as candidate)", async () => {
    const { c } = setup();
    await c.accept(
      asMessage({
        type: "system",
        subtype: "session_state_changed",
        state: "running",
      }),
    );
    expect(c.inspect().bufferSize).toBe(1);
  });

  it("emits followup with an empty buffer if the result arrives standalone", async () => {
    const { c, emit } = setup();
    await c.accept(
      asMessage<Partial<SDKResultMessage>>({
        type: "result",
        origin: { kind: "task-notification" },
      }),
    );
    expect(emit).toHaveBeenCalledTimes(1);
    expect((emit.mock.calls[0][0] as SDKMessage[]).length).toBe(0);
  });

  it("drops oldest entry and logs error when buffer cap is reached", async () => {
    const { c, logger } = setup();
    for (let i = 0; i < 257; i++) {
      await c.accept(asMessage({ type: "assistant", n: i }));
    }
    expect(c.inspect().bufferSize).toBe(256);
    expect(logger.error).toHaveBeenCalledWith(
      expect.stringContaining("off-turn followup buffer cap reached"),
    );
  });

  it("swallows and logs emitFollowup errors without throwing to the reader", async () => {
    const failingEmit = vi.fn().mockRejectedValue(new Error("network down"));
    const { c, logger } = setup(failingEmit);
    await c.accept(asMessage({ type: "assistant", n: 1 }));
    await expect(
      c.accept(
        asMessage<Partial<SDKResultMessage>>({
          type: "result",
          origin: { kind: "task-notification" },
        }),
      ),
    ).resolves.toBeUndefined();
    expect(logger.error).toHaveBeenCalledWith(
      expect.stringContaining("emitFollowup failed"),
      expect.any(Error),
    );
    expect(c.inspect()).toEqual({ state: "idle", bufferSize: 0 });
  });

  it("reset() clears any buffered candidates", async () => {
    const { c } = setup();
    await c.accept(asMessage({ type: "assistant" }));
    await c.accept(asMessage({ type: "stream_event" }));
    c.reset();
    expect(c.inspect()).toEqual({ state: "idle", bufferSize: 0 });
  });
});
