// Utilities that back the single-consumer session reader introduced for
// issue #336. Three building blocks live here so they can be unit-tested in
// isolation from acp-agent.ts:
//
//   - TurnQueue: single-producer / single-consumer async queue used by the
//     reader to hand in-turn SDK messages to the active prompt() loop.
//   - classifyOffTurn: pure classification of an off-turn message into either
//     a task-lifecycle event (emit immediately) or a followup candidate
//     (feed to the collector).
//   - OffTurnFollowupCollector: tiny state machine that accumulates followup
//     candidates between turns and decides at the closing result whether to
//     forward them as an autonomous followup or discard them as aftermath.
import type { SDKMessage, SDKResultMessage } from "@anthropic-ai/claude-agent-sdk";
import type { Logger } from "./acp-agent.js";

/** Single-producer / single-consumer async queue. The session reader is the
 *  sole producer; prompt() is the sole consumer per turn. Errors raised on
 *  the producer side (e.g. the SDK iterator throwing) propagate to whichever
 *  side reads next so the consumer can surface them as a request error. */
export class TurnQueue {
  private buffer: SDKMessage[] = [];
  private waiter: ((r: IteratorResult<SDKMessage>) => void) | null = null;
  private waiterReject: ((err: unknown) => void) | null = null;
  private err: unknown = undefined;
  private hasError = false;
  private closed = false;

  /** Push a message. If a consumer is waiting, resolve it directly. */
  push(msg: SDKMessage): void {
    if (this.closed || this.hasError) {
      // Drop silently: queue is in a terminal state. This branch only fires
      // if the reader keeps producing after close/error, which would be a
      // bug — we choose to drop rather than throw because the reader's
      // top-level catch already logs.
      return;
    }
    if (this.waiter) {
      const w = this.waiter;
      this.waiter = null;
      this.waiterReject = null;
      w({ value: msg, done: false });
      return;
    }
    this.buffer.push(msg);
  }

  /** Mark the queue closed. A pending consumer is resolved with done:true.
   *  Subsequent take() calls return done:true. Idempotent. */
  close(): void {
    if (this.closed || this.hasError) return;
    this.closed = true;
    if (this.waiter) {
      const w = this.waiter;
      this.waiter = null;
      this.waiterReject = null;
      w({ value: undefined as unknown as SDKMessage, done: true });
    }
  }

  /** Drop buffered messages that were read for a turn the consumer has
   *  abandoned. Does not resolve/reject a pending take(). */
  clear(): void {
    this.buffer = [];
  }

  /** Push an error to the consumer. A pending consumer is rejected. */
  error(err: unknown): void {
    if (this.closed || this.hasError) return;
    this.hasError = true;
    this.err = err;
    if (this.waiterReject) {
      const r = this.waiterReject;
      this.waiter = null;
      this.waiterReject = null;
      r(err);
    }
  }

  /** Consume the next message. Single-consumer: throws if a previous take()
   *  is still pending. */
  take(): Promise<IteratorResult<SDKMessage>> {
    if (this.waiter) {
      throw new Error("TurnQueue: concurrent take() is not supported");
    }
    if (this.buffer.length > 0) {
      const value = this.buffer.shift()!;
      return Promise.resolve({ value, done: false });
    }
    if (this.hasError) {
      return Promise.reject(this.err);
    }
    if (this.closed) {
      return Promise.resolve({ value: undefined as unknown as SDKMessage, done: true });
    }
    return new Promise<IteratorResult<SDKMessage>>((resolve, reject) => {
      this.waiter = resolve;
      this.waiterReject = reject;
    });
  }

  /** Test/diagnostic: number of buffered messages awaiting take(). */
  size(): number {
    return this.buffer.length;
  }
}

/** Classify a message the reader sees while no prompt() is active. Pure
 *  function for testability and to keep the off-turn policy in one place. */
export type OffTurnClassification = "lifecycle" | "followup-candidate";

export function classifyOffTurn(msg: SDKMessage): OffTurnClassification {
  if (
    msg.type === "system" &&
    (msg.subtype === "task_started" || msg.subtype === "task_notification")
  ) {
    return "lifecycle";
  }
  return "followup-candidate";
}

/** Soft cap on the followup-candidate buffer. The collector only buffers
 *  off-turn messages until the SDK emits the closing `result` or `idle` of
 *  the followup, so in normal operation the buffer stays small. The cap
 *  exists to keep a misbehaving SDK (followup that never closes) from
 *  growing memory without bound. Reaching it is logged as an error so we
 *  notice. */
const OFF_TURN_BUFFER_CAP = 256;

export type FollowupEmitter = (msgs: SDKMessage[], result: SDKResultMessage) => Promise<void>;

/** Mini state machine for messages that arrive while no prompt() is active.
 *
 *  States:
 *    idle       — no buffered candidates
 *    collecting — at least one candidate buffered; waiting for the closing
 *                 result or session_state_changed:idle
 *
 *  Transitions:
 *    any → result(origin=task-notification): emit followup, reset
 *    any → result(other origin or none):     discard with log, reset
 *    any → session_state_changed:idle:       discard with log if buffered, reset
 *    any → otherwise:                        accumulate, state=collecting
 *
 *  Lifecycle messages (task_started / task_notification) never reach the
 *  collector — the reader emits them directly. */
export class OffTurnFollowupCollector {
  private state: "idle" | "collecting" = "idle";
  private buffer: SDKMessage[] = [];

  constructor(
    private readonly sessionId: string,
    private readonly emitFollowup: FollowupEmitter,
    private readonly logger: Logger,
  ) {}

  /** Process one off-turn message. Lifecycle messages must be filtered by
   *  the caller before reaching this method. */
  async accept(msg: SDKMessage): Promise<void> {
    if (msg.type === "result") {
      const origin = (msg as SDKResultMessage).origin;
      const isFollowup = origin?.kind === "task-notification";
      const buffered = this.buffer;
      this.reset();
      if (isFollowup) {
        try {
          await this.emitFollowup(buffered, msg as SDKResultMessage);
        } catch (err) {
          this.logger.error(`Session ${this.sessionId}: emitFollowup failed:`, err);
        }
      } else {
        if (buffered.length > 0) {
          this.logger.log(
            `Session ${this.sessionId}: discarding ${buffered.length} off-turn messages followed by non-followup result`,
          );
        }
      }
      return;
    }

    if (msg.type === "system" && msg.subtype === "session_state_changed" && msg.state === "idle") {
      if (this.buffer.length > 0) {
        this.logger.log(
          `Session ${this.sessionId}: discarding ${this.buffer.length} off-turn messages with no closing result`,
        );
      }
      this.reset();
      return;
    }

    if (this.buffer.length >= OFF_TURN_BUFFER_CAP) {
      const dropped = this.buffer.shift();
      this.logger.error(
        `Session ${this.sessionId}: off-turn followup buffer cap reached (${OFF_TURN_BUFFER_CAP}); dropping oldest ${dropped?.type}`,
      );
    }
    this.state = "collecting";
    this.buffer.push(msg);
  }

  /** Discard any buffered followup-candidate messages. Used on teardown and
   *  when the reader exits. */
  reset(): void {
    this.buffer = [];
    this.state = "idle";
  }

  /** Diagnostic: current state for tests. */
  inspect(): { state: "idle" | "collecting"; bufferSize: number } {
    return { state: this.state, bufferSize: this.buffer.length };
  }
}
