import { describe, it, expect, beforeAll, beforeEach, afterAll, vi } from "vitest";
import { spawn, spawnSync } from "child_process";
import {
  Agent,
  AgentSideConnection,
  AvailableCommand,
  Client,
  ClientSideConnection,
  CreateElicitationRequest,
  CreateElicitationResponse,
  ndJsonStream,
  NewSessionResponse,
  ReadTextFileRequest,
  ReadTextFileResponse,
  RequestPermissionRequest,
  RequestPermissionResponse,
  SessionNotification,
  WriteTextFileRequest,
  WriteTextFileResponse,
} from "@agentclientprotocol/sdk";
import { nodeToWebWritable, nodeToWebReadable } from "../utils.js";
import {
  markdownEscape,
  toolInfoFromToolUse,
  toDisplayPath,
  toolUpdateFromToolResult,
  toolUpdateFromDiffToolResponse,
} from "../tools.js";
import {
  toAcpNotifications,
  promptToClaude,
  isLocalCommandMetadata,
  stripLocalCommandMetadata,
  ClaudeAcpAgent,
  claudeCliPath,
  describeAlwaysAllow,
  streamEventToAcpNotifications,
  messageIdForGrouping,
  computeFollowupUsageUpdate,
  type SDKMessageFilter,
} from "../acp-agent.js";
import { Pushable } from "../utils.js";
import { TurnQueue, OffTurnFollowupCollector } from "../session-reader.js";
import {
  deleteSession,
  getSessionMessages,
  query,
  SDKAssistantMessage,
} from "@anthropic-ai/claude-agent-sdk";
import { randomUUID } from "crypto";

/** Test helper: arrange the per-session reader for an inline-injected
 *  Session so prompt() has a producer feeding its turnQueue. Mirrors what
 *  createSession() does in production. */
function startReaderForTest(agent: ClaudeAcpAgent, sessionId: string): void {
  const session = agent.sessions[sessionId];
  if (!session) throw new Error(`startReaderForTest: session ${sessionId} not found`);
  let resolveDone!: () => void;
  session.readerDone = new Promise<void>((r) => {
    resolveDone = r;
  });
  void (
    agent as unknown as {
      startSessionReaderForTest(id: string, done: () => void): Promise<void>;
    }
  )
    .startSessionReaderForTest(sessionId, resolveDone)
    .catch(() => resolveDone());
}

vi.mock("@anthropic-ai/claude-agent-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@anthropic-ai/claude-agent-sdk")>();
  return {
    ...actual,
    deleteSession: vi.fn(),
  };
});
import type {
  BetaToolResultBlockParam,
  BetaToolSearchToolResultBlockParam,
  BetaWebSearchToolResultBlockParam,
  BetaWebFetchToolResultBlockParam,
  BetaCodeExecutionToolResultBlockParam,
} from "@anthropic-ai/sdk/resources/beta.mjs";

describe.skipIf(!process.env.RUN_INTEGRATION_TESTS)("ACP subprocess integration", () => {
  let child: ReturnType<typeof spawn>;

  beforeAll(async () => {
    const valid = spawnSync("tsc", { stdio: "inherit" });
    if (valid.status) {
      throw new Error("failed to compile");
    }
    // Start the subprocess
    child = spawn("npm", ["run", "--silent", "dev"], {
      stdio: ["pipe", "pipe", "inherit"],
      env: process.env,
    });
    child.on("error", (error) => {
      console.error("Error starting subprocess:", error);
    });
    child.on("exit", (exit) => {
      console.error("Exited with", exit);
    });
  });

  afterAll(() => {
    child.kill();
  });

  class TestClient implements Client {
    agent: Agent;
    files: Map<string, string> = new Map();
    receivedText: string = "";
    // Records for the AskUserQuestion elicitation test.
    elicitations: CreateElicitationRequest[] = [];
    permissionToolInputs: unknown[] = [];
    chosenAnswers: Record<string, string | string[]> = {};
    resolveAvailableCommands: (commands: AvailableCommand[]) => void;
    availableCommandsPromise: Promise<AvailableCommand[]>;

    constructor(agent: Agent) {
      this.agent = agent;
      this.resolveAvailableCommands = () => {};
      this.availableCommandsPromise = new Promise((resolve) => {
        this.resolveAvailableCommands = resolve;
      });
    }

    takeReceivedText() {
      const text = this.receivedText;
      this.receivedText = "";
      return text;
    }

    async requestPermission(params: RequestPermissionRequest): Promise<RequestPermissionResponse> {
      // Record what asked for permission so a test can assert that
      // AskUserQuestion did NOT fall back to a generic permission prompt.
      this.permissionToolInputs.push(params.toolCall?.rawInput);
      const optionId = params.options.find((p) => p.kind === "allow_once")!.optionId;

      return { outcome: { outcome: "selected", optionId } };
    }

    async unstable_createElicitation(
      params: CreateElicitationRequest,
    ): Promise<CreateElicitationResponse> {
      this.elicitations.push(params);
      if (params.mode !== "form") {
        return { action: "decline" };
      }
      // Accept the first option of every choice field (skip the free-text one).
      const content: Record<string, string | string[]> = {};
      for (const [key, prop] of Object.entries(params.requestedSchema.properties ?? {})) {
        if (key === "customAnswer") continue;
        const p = prop as {
          oneOf?: Array<{ const: string }>;
          items?: { anyOf?: Array<{ const: string }> };
        };
        if (p.oneOf?.length) {
          content[key] = p.oneOf[0].const;
        } else if (p.items?.anyOf?.length) {
          content[key] = [p.items.anyOf[0].const];
        }
      }
      this.chosenAnswers = content;
      return { action: "accept", content };
    }

    async sessionUpdate(params: SessionNotification): Promise<void> {
      console.error("RECEIVED", JSON.stringify(params, null, 4));

      switch (params.update.sessionUpdate) {
        case "agent_message_chunk": {
          if (params.update.content.type === "text") {
            this.receivedText += params.update.content.text;
          }
          break;
        }
        case "available_commands_update":
          this.resolveAvailableCommands(params.update.availableCommands);
          break;
        default:
          break;
      }
    }

    async writeTextFile(params: WriteTextFileRequest): Promise<WriteTextFileResponse> {
      this.files.set(params.path, params.content);
      return {};
    }

    async readTextFile(params: ReadTextFileRequest): Promise<ReadTextFileResponse> {
      const content = this.files.get(params.path) ?? "";
      return {
        content,
      };
    }
  }

  async function setupTestSession(cwd: string): Promise<{
    client: TestClient;
    connection: ClientSideConnection;
    newSessionResponse: NewSessionResponse;
  }> {
    let client;
    const input = nodeToWebWritable(child.stdin!);
    const output = nodeToWebReadable(child.stdout!);
    const stream = ndJsonStream(input, output);
    const connection = new ClientSideConnection((agent) => {
      client = new TestClient(agent);
      return client;
    }, stream);

    await connection.initialize({
      protocolVersion: 1,
      clientCapabilities: {
        fs: {
          readTextFile: true,
          writeTextFile: true,
        },
        elicitation: {
          form: {},
        },
      },
    });

    const newSessionResponse = await connection.newSession({
      cwd,
      mcpServers: [],
    });

    return { client: client!, connection, newSessionResponse };
  }

  it("should connect to the ACP subprocess", async () => {
    const { client, connection, newSessionResponse } = await setupTestSession(process.cwd());

    await connection.prompt({
      prompt: [
        {
          type: "text",
          text: "Hello",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).not.toEqual("");
  }, 30000);

  it("should include available commands", async () => {
    const { client, connection, newSessionResponse } = await setupTestSession(__dirname);

    const commands = await client.availableCommandsPromise;

    expect(commands).toContainEqual({
      name: "quick-math",
      description: "10 * 3 = 30 (project)",
      input: null,
    });
    expect(commands).toContainEqual({
      name: "say-hello",
      description: "Say hello (project)",
      input: { hint: "name" },
    });

    await connection.prompt({
      prompt: [
        {
          type: "text",
          text: "/quick-math",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).toContain("30");

    await connection.prompt({
      prompt: [
        {
          type: "text",
          text: "/say-hello GPT-5",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).toContain("Hello GPT-5");
  }, 30000);

  it("/compact works", async () => {
    const { client, connection, newSessionResponse } = await setupTestSession(__dirname);

    const commands = await client.availableCommandsPromise;

    expect(commands).toContainEqual({
      description: "Free up context by summarizing the conversation so far",
      input: {
        hint: "<optional custom summarization instructions>",
      },
      name: "compact",
    });

    // Build up enough conversation that there's something to compact. The SDK
    // refuses to compact a conversation with too few message groups.
    for (let i = 0; i < 6; i++) {
      await connection.prompt({
        prompt: [{ type: "text", text: `Reply with just the number ${i}.` }],
        sessionId: newSessionResponse.sessionId,
      });
      client.takeReceivedText();
    }

    await connection.prompt({
      prompt: [
        {
          type: "text",
          text: "/compact",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    expect(client.takeReceivedText()).toContain("Compacting...\n\nCompacting completed.");
  }, 60000);

  // Regression guard for the SDK's AskUserQuestion routing. The built-in
  // AskUserQuestion tool is delivered to us through `canUseTool` (not the
  // interactive `onUserDialog` path), where we intercept it and render an ACP
  // form elicitation, returning the answer via `updatedInput`. If a future SDK
  // changes that routing — e.g. stops calling `canUseTool` for it, or no longer
  // reads answers back from `updatedInput` — this test fails: either no
  // elicitation arrives, the tool falls back to a permission prompt, or the
  // answer never reaches the model's reply.
  it("routes AskUserQuestion through ACP form elicitation and round-trips the answer", async () => {
    const { client, connection, newSessionResponse } = await setupTestSession(process.cwd());

    await connection.prompt({
      prompt: [
        {
          type: "text",
          text:
            "Use the AskUserQuestion tool right now to ask me to choose a favorite color. " +
            "Offer exactly two options: 'Red' and 'Blue'. Do not use any other tool and do " +
            "not ask in plain text. After I answer, reply with one short sentence naming the " +
            "color I picked.",
        },
      ],
      sessionId: newSessionResponse.sessionId,
    });

    // The tool surfaced as an ACP form elicitation...
    expect(client.elicitations.length).toBeGreaterThan(0);
    const elicitation = client.elicitations[0];
    expect(elicitation.mode).toBe("form");

    // ...built by our converter (indexed field key + free-text "Other" field),
    // which confirms our interception path produced it rather than some other
    // mechanism.
    const properties =
      elicitation.mode === "form" ? Object.keys(elicitation.requestedSchema.properties ?? {}) : [];
    expect(properties).toContain("question_0");
    expect(properties).toContain("customAnswer");

    // AskUserQuestion must NOT fall back to a generic permission prompt: no
    // permission request should have carried AskUserQuestion's `questions`.
    const fellBackToPermission = client.permissionToolInputs.some(
      (input) =>
        !!input &&
        typeof input === "object" &&
        Array.isArray((input as { questions?: unknown }).questions),
    );
    expect(fellBackToPermission).toBe(false);

    // The chosen answer round-trips: the model's reply names the picked option.
    const picked = String(Object.values(client.chosenAnswers)[0] ?? "");
    expect(picked).not.toEqual("");
    expect(client.takeReceivedText().toLowerCase()).toContain(picked.toLowerCase());
  }, 60000);
});

describe("tool conversions", () => {
  it("should handle Bash nicely", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01VtsS2mxUFwpBJZYd7BmbC9",
      name: "Bash",
      input: {
        command: "rm README.md.rm",
        description: "Delete README.md.rm file",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "execute",
      title: "rm README.md.rm",
      content: [
        {
          content: {
            text: "Delete README.md.rm file",
            type: "text",
          },
          type: "content",
        },
      ],
    });
  });

  it("should handle Glob nicely", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01VtsS2mxUFwpBJZYd7BmbC9",
      name: "Glob",
      input: {
        pattern: "*/**.ts",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "search",
      title: "Find `*/**.ts`",
      content: [],
      locations: [],
    });
  });

  it("should handle Task tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01ANYHYDsXcDPKgxhg7us9bj",
      name: "Task",
      input: {
        description: "Handle user's work request",
        prompt:
          'The user has asked me to "Create a Task to do the work!" but hasn\'t specified what specific work they want done. I need to:\n\n1. First understand what work needs to be done by examining the current state of the repository\n2. Look at the git status to see what files have been modified\n3. Check if there are any obvious tasks that need completion based on the current state\n4. If the work isn\'t clear from the context, ask the user to specify what work they want accomplished\n\nThe git status shows: "M src/tests/acp-agent.test.ts" - there\'s a modified test file that might need attention.\n\nPlease examine the repository state and determine what work needs to be done, then either complete it or ask the user for clarification on the specific task they want accomplished.',
        subagent_type: "general-purpose",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "think",
      title: "Handle user's work request",
      content: [
        {
          content: {
            text: 'The user has asked me to "Create a Task to do the work!" but hasn\'t specified what specific work they want done. I need to:\n\n1. First understand what work needs to be done by examining the current state of the repository\n2. Look at the git status to see what files have been modified\n3. Check if there are any obvious tasks that need completion based on the current state\n4. If the work isn\'t clear from the context, ask the user to specify what work they want accomplished\n\nThe git status shows: "M src/tests/acp-agent.test.ts" - there\'s a modified test file that might need attention.\n\nPlease examine the repository state and determine what work needs to be done, then either complete it or ask the user for clarification on the specific task they want accomplished.',
            type: "text",
          },
          type: "content",
        },
      ],
    });
  });

  it("should handle Grep tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_016j8oGSD3eAZ9KT62Y7Jsjb",
      name: "Grep",
      input: {
        pattern: ".*",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "search",
      title: 'grep ".*"',
      content: [],
    });
  });

  it("should handle Write tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01ABC123XYZ789",
      name: "Write",
      input: {
        file_path: "/Users/test/project/example.txt",
        content: "Hello, World!\nThis is test content.",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "edit",
      title: "Write /Users/test/project/example.txt",
      content: [
        {
          type: "diff",
          path: "/Users/test/project/example.txt",
          oldText: null,
          newText: "Hello, World!\nThis is test content.",
        },
      ],
      locations: [{ path: "/Users/test/project/example.txt" }],
    });
  });

  it("should handle Write tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01GHI789JKL456",
      name: "Write",
      input: {
        file_path: "/Users/test/project/config.json",
        content: '{"version": "1.0.0"}',
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "edit",
      title: "Write /Users/test/project/config.json",
      content: [
        {
          type: "diff",
          path: "/Users/test/project/config.json",
          oldText: null,
          newText: '{"version": "1.0.0"}',
        },
      ],
      locations: [{ path: "/Users/test/project/config.json" }],
    });
  });

  it("should handle Edit tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01EDIT123",
      name: "Edit",
      input: {
        file_path: "/Users/test/project/test.txt",
        old_string: "old text",
        new_string: "new text",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "edit",
      title: "Edit /Users/test/project/test.txt",
      content: [
        {
          type: "diff",
          path: "/Users/test/project/test.txt",
          oldText: "old text",
          newText: "new text",
        },
      ],
      locations: [{ path: "/Users/test/project/test.txt" }],
    });
  });

  it("should handle Edit tool calls with replace_all", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01EDIT456",
      name: "Edit",
      input: {
        replace_all: false,
        file_path: "/Users/benbrandt/github/codex-acp/src/thread.rs",
        old_string:
          "struct PromptState {\n    active_command: Option<ActiveCommand>,\n    active_web_search: Option<String>,\n}",
        new_string:
          "struct PromptState {\n    active_commands: HashMap<String, ActiveCommand>,\n    active_web_search: Option<String>,\n}",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "edit",
      title: "Edit /Users/benbrandt/github/codex-acp/src/thread.rs",
      content: [
        {
          type: "diff",
          path: "/Users/benbrandt/github/codex-acp/src/thread.rs",
          oldText:
            "struct PromptState {\n    active_command: Option<ActiveCommand>,\n    active_web_search: Option<String>,\n}",
          newText:
            "struct PromptState {\n    active_commands: HashMap<String, ActiveCommand>,\n    active_web_search: Option<String>,\n}",
        },
      ],
      locations: [{ path: "/Users/benbrandt/github/codex-acp/src/thread.rs" }],
    });
  });

  it("should handle Edit tool calls without file_path", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01EDIT789",
      name: "Edit",
      input: {},
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "edit",
      title: "Edit",
      content: [],
      locations: [],
    });
  });

  it("should handle Read tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01MNO456PQR789",
      name: "Read",
      input: {
        file_path: "/Users/test/project/readme.md",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/readme.md",
      content: [],
      locations: [{ path: "/Users/test/project/readme.md", line: 1 }],
    });
  });

  it("should handle Read tool calls", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01YZA789BCD123",
      name: "Read",
      input: {
        file_path: "/Users/test/project/data.json",
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/data.json",
      content: [],
      locations: [{ path: "/Users/test/project/data.json", line: 1 }],
    });
  });

  it("should handle Read with limit", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01EFG456HIJ789",
      name: "Read",
      input: {
        file_path: "/Users/test/project/large.txt",
        limit: 100,
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/large.txt (1 - 100)",
      content: [],
      locations: [{ path: "/Users/test/project/large.txt", line: 1 }],
    });
  });

  it("should handle Read with offset and limit", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01KLM789NOP456",
      name: "Read",
      input: {
        file_path: "/Users/test/project/large.txt",
        offset: 50,
        limit: 100,
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/large.txt (50 - 149)",
      content: [],
      locations: [{ path: "/Users/test/project/large.txt", line: 50 }],
    });
  });

  it("should handle Read with only offset", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01QRS123TUV789",
      name: "Read",
      input: {
        file_path: "/Users/test/project/large.txt",
        offset: 200,
      },
    };

    expect(toolInfoFromToolUse(tool_use)).toStrictEqual({
      kind: "read",
      title: "Read /Users/test/project/large.txt (from line 200)",
      content: [],
      locations: [{ path: "/Users/test/project/large.txt", line: 200 }],
    });
  });

  it("should use relative path in title when cwd is provided", () => {
    const tool_use = {
      type: "tool_use",
      id: "toolu_01READ_CWD",
      name: "Read",
      input: { file_path: "/Users/test/project/src/main.ts" },
    };

    const result = toolInfoFromToolUse(tool_use, false, "/Users/test/project");
    expect(result.title).toBe("Read src/main.ts");
    // locations.path stays absolute for navigation
    expect(result.locations).toStrictEqual([{ path: "/Users/test/project/src/main.ts", line: 1 }]);
  });

  it("should handle plan entries", () => {
    const received: SDKAssistantMessage = {
      type: "assistant",
      message: {
        id: "msg_017eNosJgww7F5qD4a8BcAcx",
        type: "message",
        role: "assistant",
        container: null,
        model: "claude-sonnet-4-20250514",
        content: [
          {
            type: "tool_use",
            id: "toolu_01HaXZ4LfdchSeSR8ygt4zyq",
            name: "TodoWrite",
            input: {
              todos: [
                {
                  content: "Analyze existing test coverage and identify gaps",
                  status: "in_progress",
                  activeForm: "Analyzing existing test coverage",
                },
                {
                  content: "Add comprehensive edge case tests",
                  status: "pending",
                  activeForm: "Adding comprehensive edge case tests",
                },
                {
                  content: "Add performance and timing tests",
                  status: "pending",
                  activeForm: "Adding performance and timing tests",
                },
                {
                  content: "Add error handling and panic behavior tests",
                  status: "pending",
                  activeForm: "Adding error handling tests",
                },
                {
                  content: "Add concurrent access and race condition tests",
                  status: "pending",
                  activeForm: "Adding concurrent access tests",
                },
                {
                  content: "Add tests for Each function with various data types",
                  status: "pending",
                  activeForm: "Adding Each function tests",
                },
                {
                  content: "Add benchmark tests for performance measurement",
                  status: "pending",
                  activeForm: "Adding benchmark tests",
                },
                {
                  content: "Improve test organization and helper functions",
                  status: "pending",
                  activeForm: "Improving test organization",
                },
              ],
            },
          },
        ],
        stop_reason: null,
        stop_sequence: null,
        stop_details: null,
        diagnostics: null,
        usage: {
          input_tokens: 6,
          cache_creation_input_tokens: 326,
          cache_read_input_tokens: 17265,
          cache_creation: {
            ephemeral_5m_input_tokens: 326,
            ephemeral_1h_input_tokens: 0,
          },
          output_tokens: 1,
          service_tier: "standard",
          server_tool_use: null,
          inference_geo: null,
          iterations: null,
          output_tokens_details: null,
          speed: null,
        },
        context_management: null,
      },
      parent_tool_use_id: null,
      session_id: "d056596f-e328-41e9-badd-b07122ae5227",
      uuid: "b7c3330c-de8f-4bba-ac53-68c7f76ffeb5",
    };
    expect(
      toAcpNotifications(
        received.message.content,
        received.message.role,
        "test",
        {},
        {} as AgentSideConnection,
        console,
      ),
    ).toStrictEqual([
      {
        sessionId: "test",
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Analyze existing test coverage and identify gaps",
              priority: "medium",
              status: "in_progress",
            },
            {
              content: "Add comprehensive edge case tests",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add performance and timing tests",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add error handling and panic behavior tests",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add concurrent access and race condition tests",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add tests for Each function with various data types",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Add benchmark tests for performance measurement",
              priority: "medium",
              status: "pending",
            },
            {
              content: "Improve test organization and helper functions",
              priority: "medium",
              status: "pending",
            },
          ],
        },
      },
    ]);
  });

  it("should return empty update for successful edit result", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "Edit",
      input: {
        file_path: "/Users/test/project/test.txt",
        old_string: "old",
        new_string: "new",
      },
    };

    const toolResult = {
      content: [
        {
          type: "text" as const,
          text: "not valid json",
        },
      ],
      tool_use_id: "test",
      is_error: false,
      type: "tool_result" as const,
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    // Should return empty object when parsing fails
    expect(update).toEqual({});
  });

  it("should return content update for edit failure", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "Edit",
      input: {
        file_path: "/Users/test/project/test.txt",
        old_string: "old",
        new_string: "new",
      },
    };

    const toolResult = {
      content: [
        {
          type: "text" as const,
          text: "Failed to find `old_string`",
        },
      ],
      tool_use_id: "test",
      is_error: true,
      type: "tool_result" as const,
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    // Should return empty object when parsing fails
    expect(update).toEqual({
      content: [
        {
          content: { type: "text", text: "```\nFailed to find `old_string`\n```" },
          type: "content",
        },
      ],
    });
  });

  it("should transform tool_reference content to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "ToolSearch",
      input: { query: "test" },
    };

    const toolResult: BetaToolResultBlockParam = {
      content: [
        {
          type: "tool_reference",
          tool_name: "some_discovered_tool",
        },
      ],
      tool_use_id: "toolu_01MNO345",
      is_error: false,
      type: "tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Tool: some_discovered_tool" },
        },
      ],
    });
  });

  it("should transform web_search_result content to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "WebSearch",
      input: { query: "test" },
    };

    const toolResult: BetaWebSearchToolResultBlockParam = {
      content: [
        {
          type: "web_search_result",
          title: "Test Result",
          url: "https://example.com",
          encrypted_content: "...",
          page_age: null,
        },
      ],
      tool_use_id: "toolu_01MNO345",
      type: "web_search_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Test Result (https://example.com)" },
        },
      ],
    });
  });

  it("should transform web_search_tool_result_error to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "WebSearch",
      input: { query: "test" },
    };

    const toolResult: BetaWebSearchToolResultBlockParam = {
      content: {
        type: "web_search_tool_result_error",
        error_code: "unavailable",
      },
      tool_use_id: "toolu_01MNO345",
      type: "web_search_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Error: unavailable" },
        },
      ],
    });
  });

  it("should transform code_execution_result content to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "CodeExecution",
      input: {},
    };

    const toolResult: BetaCodeExecutionToolResultBlockParam = {
      content: {
        type: "code_execution_result",
        stdout: "Hello World",
        stderr: "",
        return_code: 0,
        content: [],
      },
      tool_use_id: "toolu_01MNO345",
      type: "code_execution_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Output: Hello World" },
        },
      ],
    });
  });

  it("should transform web_fetch_result content to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "WebFetch",
      input: { url: "https://example.com" },
    };

    const toolResult: BetaWebFetchToolResultBlockParam = {
      content: {
        type: "web_fetch_result",
        url: "https://example.com",
        content: {
          type: "document",
          citations: null,
          title: null,
          source: { type: "text", media_type: "text/plain", data: "Page content here" },
        },
      },
      tool_use_id: "toolu_01MNO345",
      type: "web_fetch_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Fetched: https://example.com" },
        },
      ],
    });
  });

  it("should transform tool_search_tool_search_result to valid ACP content", () => {
    const toolUse = {
      type: "tool_use",
      id: "toolu_01MNO345",
      name: "ToolSearch",
      input: { query: "test" },
    };

    const toolResult: BetaToolSearchToolResultBlockParam = {
      content: {
        type: "tool_search_tool_search_result",
        tool_references: [
          { type: "tool_reference", tool_name: "tool_a" },
          { type: "tool_reference", tool_name: "tool_b" },
        ],
      },
      tool_use_id: "toolu_01MNO345",
      type: "tool_search_tool_result",
    };

    const update = toolUpdateFromToolResult(toolResult, toolUse);

    expect(update).toEqual({
      content: [
        {
          type: "content",
          content: { type: "text", text: "Tools found: tool_a, tool_b" },
        },
      ],
    });
  });
});

describe("toDisplayPath", () => {
  it("should relativize paths inside cwd and keep absolute paths outside", () => {
    expect(toDisplayPath("/Users/test/project/src/main.ts", "/Users/test/project")).toBe(
      "src/main.ts",
    );
    expect(toDisplayPath("/etc/hosts", "/Users/test/project")).toBe("/etc/hosts");
    expect(toDisplayPath("/Users/test/project/src/main.ts")).toBe(
      "/Users/test/project/src/main.ts",
    );
    // Partial directory name match should not be treated as inside cwd
    expect(toDisplayPath("/Users/test/project-other/file.ts", "/Users/test/project")).toBe(
      "/Users/test/project-other/file.ts",
    );
  });
});

describe("toolUpdateFromDiffToolResponse", () => {
  it("should return empty for non-object input", () => {
    expect(toolUpdateFromDiffToolResponse(null)).toEqual({});
    expect(toolUpdateFromDiffToolResponse(undefined)).toEqual({});
    expect(toolUpdateFromDiffToolResponse("string")).toEqual({});
  });

  it("should return empty when filePath or structuredPatch is missing", () => {
    expect(toolUpdateFromDiffToolResponse({})).toEqual({});
    expect(toolUpdateFromDiffToolResponse({ filePath: "/foo.ts" })).toEqual({});
    expect(toolUpdateFromDiffToolResponse({ structuredPatch: [] })).toEqual({});
  });

  it("should build diff content from a single-hunk structuredPatch", () => {
    const toolResponse = {
      filePath: "/Users/test/project/test.txt",
      structuredPatch: [
        {
          oldStart: 1,
          oldLines: 3,
          newStart: 1,
          newLines: 3,
          lines: [" context before", "-old line", "+new line", " context after"],
        },
      ],
    };

    expect(toolUpdateFromDiffToolResponse(toolResponse)).toEqual({
      content: [
        {
          type: "diff",
          path: "/Users/test/project/test.txt",
          oldText: "context before\nold line\ncontext after",
          newText: "context before\nnew line\ncontext after",
        },
      ],
      locations: [{ path: "/Users/test/project/test.txt", line: 1 }],
    });
  });

  it("should build multiple diff content blocks for replaceAll with multiple hunks", () => {
    const toolResponse = {
      filePath: "/Users/test/project/file.ts",
      structuredPatch: [
        {
          oldStart: 5,
          oldLines: 1,
          newStart: 5,
          newLines: 1,
          lines: ["-oldValue", "+newValue"],
        },
        {
          oldStart: 20,
          oldLines: 1,
          newStart: 20,
          newLines: 1,
          lines: ["-oldValue", "+newValue"],
        },
      ],
    };

    expect(toolUpdateFromDiffToolResponse(toolResponse)).toEqual({
      content: [
        {
          type: "diff",
          path: "/Users/test/project/file.ts",
          oldText: "oldValue",
          newText: "newValue",
        },
        {
          type: "diff",
          path: "/Users/test/project/file.ts",
          oldText: "oldValue",
          newText: "newValue",
        },
      ],
      locations: [
        { path: "/Users/test/project/file.ts", line: 5 },
        { path: "/Users/test/project/file.ts", line: 20 },
      ],
    });
  });

  it("should handle deletion (newText becomes empty string)", () => {
    const toolResponse = {
      filePath: "/Users/test/project/file.ts",
      structuredPatch: [
        {
          oldStart: 10,
          oldLines: 2,
          newStart: 10,
          newLines: 1,
          lines: [" context", "-removed line"],
        },
      ],
    };

    expect(toolUpdateFromDiffToolResponse(toolResponse)).toEqual({
      content: [
        {
          type: "diff",
          path: "/Users/test/project/file.ts",
          oldText: "context\nremoved line",
          newText: "context",
        },
      ],
      locations: [{ path: "/Users/test/project/file.ts", line: 10 }],
    });
  });

  it("should return empty for empty structuredPatch array", () => {
    const toolResponse = {
      filePath: "/Users/test/project/file.ts",
      structuredPatch: [],
    };

    expect(toolUpdateFromDiffToolResponse(toolResponse)).toEqual({});
  });
});

describe("stripLocalCommandMetadata", () => {
  it("returns null for strings that are pure marker metadata", () => {
    expect(stripLocalCommandMetadata("<command-name>/model</command-name>")).toBeNull();
    expect(
      stripLocalCommandMetadata("<local-command-stdout>out</local-command-stdout>"),
    ).toBeNull();
    expect(
      stripLocalCommandMetadata("<local-command-stderr>err</local-command-stderr>"),
    ).toBeNull();
    expect(
      stripLocalCommandMetadata(
        "<command-name>/model</command-name>\n            <command-message>model</command-message>\n            <command-args>opus</command-args>",
      ),
    ).toBeNull();
  });

  it("returns the string unchanged for real content", () => {
    expect(stripLocalCommandMetadata("hi")).toBe("hi");
    expect(stripLocalCommandMetadata("please run /model with args")).toBe(
      "please run /model with args",
    );
  });

  // Regression: in the original bug report the entire /model preamble and
  // the user's real "hi" prompt were concatenated into a single message.
  // We want to strip the marker tags and preserve the real prose, not drop
  // the whole message.
  it("strips marker tags from mixed-content strings, preserving real prose", () => {
    const mixed =
      "<command-name>/model</command-name>\n            <command-message>model</command-message>\n            <command-args>opus</command-args>" +
      "<local-command-stdout>Set model to opus (claude-opus-4-7)</local-command-stdout>" +
      "<command-name>/model</command-name>\n            <command-message>model</command-message>\n            <command-args>opus[1m]</command-args>" +
      "<local-command-stdout>Set model to opus[1m] (claude-opus-4-7[1m])</local-command-stdout>" +
      "hi";
    const stripped = stripLocalCommandMetadata(mixed);
    expect(typeof stripped).toBe("string");
    expect(stripped as string).not.toContain("<command-name>");
    expect(stripped as string).not.toContain("<command-message>");
    expect(stripped as string).not.toContain("<command-args>");
    expect(stripped as string).not.toContain("<local-command-stdout>");
    expect((stripped as string).trimEnd()).toMatch(/hi$/);
  });

  it("drops marker-only blocks from mixed arrays, keeping real blocks", () => {
    const result = stripLocalCommandMetadata([
      { type: "text", text: "<command-name>/model</command-name>" },
      { type: "text", text: "<local-command-stdout>ok</local-command-stdout>" },
      { type: "text", text: "hi" },
    ]);
    expect(result).toEqual([{ type: "text", text: "hi" }]);
  });

  it("returns null when every block is a marker", () => {
    expect(
      stripLocalCommandMetadata([
        { type: "text", text: "<command-name>/model</command-name>" },
        { type: "text", text: "<local-command-stdout>ok</local-command-stdout>" },
      ]),
    ).toBeNull();
  });

  it("strips tags inside a text block while keeping the trailing prose", () => {
    const result = stripLocalCommandMetadata([
      {
        type: "text",
        text: "<command-name>/model</command-name><local-command-stdout>ok</local-command-stdout>hi",
      },
    ]);
    expect(result).toEqual([{ type: "text", text: "hi" }]);
  });

  it("leaves non-text blocks alone", () => {
    const image = { type: "image", source: { type: "base64", data: "", media_type: "image/png" } };
    const result = stripLocalCommandMetadata([
      { type: "text", text: "<command-name>/model</command-name>" },
      image,
    ]);
    expect(result).toEqual([image]);
  });

  it("handles null/undefined/non-container shapes", () => {
    expect(stripLocalCommandMetadata(null)).toBeNull();
    expect(stripLocalCommandMetadata(undefined)).toBeUndefined();
    expect(stripLocalCommandMetadata({ arbitrary: "object" })).toEqual({ arbitrary: "object" });
  });
});

describe("isLocalCommandMetadata", () => {
  it("is true when stripping leaves nothing", () => {
    expect(isLocalCommandMetadata("<command-name>/model</command-name>")).toBe(true);
    expect(
      isLocalCommandMetadata([{ type: "text", text: "<command-name>/model</command-name>" }]),
    ).toBe(true);
  });

  it("is false when real content survives stripping", () => {
    expect(isLocalCommandMetadata("hi")).toBe(false);
    expect(isLocalCommandMetadata("<command-name>/model</command-name>hi")).toBe(false);
    expect(
      isLocalCommandMetadata([
        { type: "text", text: "<command-name>/model</command-name>" },
        { type: "text", text: "hi" },
      ]),
    ).toBe(false);
  });
});

describe("escape markdown", () => {
  it("should escape markdown characters", () => {
    let text = "Hello *world*!";
    let escaped = markdownEscape(text);
    expect(escaped).toEqual("```\nHello *world*!\n```");

    text = "for example:\n```markdown\nHello *world*!\n```\n";
    escaped = markdownEscape(text);
    expect(escaped).toEqual("````\nfor example:\n```markdown\nHello *world*!\n```\n````");
  });
});

describe("prompt conversion", () => {
  it("should not change built-in slash commands", () => {
    const message = promptToClaude({
      sessionId: "test",
      prompt: [
        {
          type: "text",
          text: "/compact args",
        },
      ],
    });
    expect(message.message.content).toEqual([
      {
        text: "/compact args",
        type: "text",
      },
    ]);
  });

  it("should remove MCP prefix from MCP slash commands", () => {
    const message = promptToClaude({
      sessionId: "test",
      prompt: [
        {
          type: "text",
          text: "/mcp:server:name args",
        },
      ],
    });
    expect(message.message.content).toEqual([
      {
        text: "/server:name (MCP) args",
        type: "text",
      },
    ]);
  });
});

describe.skipIf(!process.env.RUN_INTEGRATION_TESTS)("SDK behavior", () => {
  it("finds vendored cli path", async () => {
    const path = await claudeCliPath();
    expect(path).toMatch(/@anthropic-ai\/claude-agent-sdk-[^/]+\/claude(\.exe)?$/);
  });

  it("query has a 'default' model", async () => {
    const q = query({ prompt: "hi" });
    const models = await q.supportedModels();
    const defaultModel = models.find((m) => m.value === "default");
    expect(defaultModel).toBeDefined();
  }, 10000);

  it("custom session id", async () => {
    const sessionId = randomUUID();
    const q = query({
      prompt: "hi",
      options: {
        systemPrompt: { type: "preset", preset: "claude_code" },
        sessionId,
        settingSources: ["user", "project", "local"],
        includePartialMessages: true,
      },
    });

    const { value } = await q.next();
    expect(value).toMatchObject({ type: "system", session_id: sessionId });
  }, 10000);

  // Pins the SDK invariant our `messageId` plumbing relies on: the Anthropic
  // API message id is available at `message_start` (before any delta), is the
  // same on the consolidated assistant message, and is recoverable from the
  // persisted transcript — so a turn keeps one stable id across streaming and
  // replay. The per-`stream_event` uuid is NOT used because it is unique per
  // event and never persisted; this test would fail if a future SDK regressed
  // any of those properties.
  it("uses the API message id as a stable anchor across streaming and replay", async () => {
    const sessionId = randomUUID();
    const q = query({
      prompt: "Reply with exactly these words and nothing else: hello there my friend",
      options: {
        systemPrompt: { type: "preset", preset: "claude_code" },
        sessionId,
        includePartialMessages: true,
        maxTurns: 1,
        allowedTools: [],
      },
    });

    let messageStartApiId: string | undefined;
    let consolidatedApiId: string | undefined;
    let sawDelta = false;
    let allPartialsTopLevel = true;

    for await (const message of q) {
      if (message.type === "assistant") {
        consolidatedApiId = message.message.id;
      }
      if (message.type !== "stream_event") continue;
      // Every streaming partial must belong to the top-level agent
      // (parent_tool_use_id === null). Subagent work is folded into tool-result
      // messages rather than surfaced as partial streams, which is what lets us
      // track a single anchor without keying by parent_tool_use_id.
      if (message.parent_tool_use_id !== null) allPartialsTopLevel = false;
      if (message.event.type === "message_start") {
        messageStartApiId = message.event.message.id;
      } else if (message.event.type === "content_block_delta") {
        sawDelta = true;
      }
    }

    // The API message id is present at message_start (before deltas), so we can
    // tag every streamed chunk with it, and it is identical on the consolidated
    // assistant message.
    expect(messageStartApiId).toBeTruthy();
    expect(sawDelta).toBe(true);
    expect(allPartialsTopLevel).toBe(true);
    expect(consolidatedApiId).toBe(messageStartApiId);

    // ...and the SAME id is recoverable from the persisted transcript, so chunks
    // grouped live keep their id when the session is replayed.
    const persisted = await getSessionMessages(sessionId);
    const replayedAssistant = persisted.find((m) => m.type === "assistant");
    expect(replayedAssistant).toBeDefined();
    expect((replayedAssistant!.message as { id?: string }).id).toBe(messageStartApiId);
    // The helper used in production must derive that same id from the replayed
    // message.
    expect(messageIdForGrouping(replayedAssistant!)).toBe(messageStartApiId);
  }, 30000);
});

describe("permission requests", () => {
  it("should include title field in tool permission request structure", () => {
    // Test various tool types to ensure title is correctly generated
    const testCases = [
      {
        toolUse: {
          type: "tool_use" as const,
          id: "test-1",
          name: "Write",
          input: { file_path: "/test/file.txt", content: "test" },
        },
        expectedTitlePart: "/test/file.txt",
      },
      {
        toolUse: {
          type: "tool_use" as const,
          id: "test-2",
          name: "Bash",
          input: { command: "ls -la", description: "List files" },
        },
        expectedTitlePart: "ls -la",
      },
      {
        toolUse: {
          type: "tool_use" as const,
          id: "test-3",
          name: "Read",
          input: { file_path: "/test/data.json" },
        },
        expectedTitlePart: "/test/data.json",
      },
    ];

    for (const testCase of testCases) {
      // Get the tool info that would be used in requestPermission
      const toolInfo = toolInfoFromToolUse(testCase.toolUse);

      // Verify toolInfo has a title
      expect(toolInfo.title).toBeDefined();
      expect(toolInfo.title).toContain(testCase.expectedTitlePart);

      // Verify the structure that our fix creates for requestPermission
      // We now spread the full toolInfo (title, kind, content, locations)
      const requestStructure = {
        toolCall: {
          toolCallId: testCase.toolUse.id,
          rawInput: testCase.toolUse.input,
          ...toolInfo,
        },
      };

      // Ensure the title field is present and populated
      expect(requestStructure.toolCall.title).toBeDefined();
      expect(requestStructure.toolCall.title).toContain(testCase.expectedTitlePart);

      // Ensure kind is included so the client can render appropriate UI
      expect(requestStructure.toolCall.kind).toBeDefined();
      expect(typeof requestStructure.toolCall.kind).toBe("string");

      // Ensure content is included so the client always has tool call details
      expect(requestStructure.toolCall.content).toBeDefined();
      expect(Array.isArray(requestStructure.toolCall.content)).toBe(true);
    }
  });

  describe("describeAlwaysAllow", () => {
    it("falls back to naming the whole tool when no suggestions are provided", () => {
      expect(describeAlwaysAllow(undefined, "Bash")).toBe("Always Allow all Bash");
      expect(describeAlwaysAllow([], "Read")).toBe("Always Allow all Read");
    });

    it("includes the scoped rule content from a suggestion", () => {
      const label = describeAlwaysAllow(
        [
          {
            type: "addRules",
            rules: [{ toolName: "Bash", ruleContent: "npm test:*" }],
            behavior: "allow",
            destination: "session",
          },
        ],
        "Bash",
      );
      expect(label).toBe("Always Allow Bash(npm test:*)");
    });

    it("indicates a tool-wide rule when the suggestion has no ruleContent", () => {
      const label = describeAlwaysAllow(
        [
          {
            type: "addRules",
            rules: [{ toolName: "Read" }],
            behavior: "allow",
            destination: "session",
          },
        ],
        "Read",
      );
      expect(label).toBe("Always Allow all Read");
    });

    it("joins multiple rules and directory suggestions", () => {
      const label = describeAlwaysAllow(
        [
          {
            type: "addRules",
            rules: [
              { toolName: "Bash", ruleContent: "git status" },
              { toolName: "Bash", ruleContent: "git diff:*" },
            ],
            behavior: "allow",
            destination: "session",
          },
          {
            type: "addDirectories",
            directories: ["/tmp/work"],
            destination: "session",
          },
        ],
        "Bash",
      );
      expect(label).toBe("Always Allow Bash(git status), Bash(git diff:*) and access to /tmp/work");
    });

    it("ignores non-allow rules and falls back when nothing is left", () => {
      const label = describeAlwaysAllow(
        [
          {
            type: "addRules",
            rules: [{ toolName: "Bash", ruleContent: "rm -rf:*" }],
            behavior: "deny",
            destination: "session",
          },
        ],
        "Bash",
      );
      expect(label).toBe("Always Allow all Bash");
    });
  });
});

describe("stop reason propagation", () => {
  function createMockAgent() {
    const mockClient = {
      sessionUpdate: async () => {},
    } as unknown as AgentSideConnection;
    return new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
  }

  function createResultMessage(overrides: {
    subtype: "success" | "error_during_execution";
    stop_reason: string | null;
    is_error: boolean;
    result?: string;
    errors?: string[];
  }) {
    return {
      type: "result" as const,
      subtype: overrides.subtype,
      stop_reason: overrides.stop_reason,
      is_error: overrides.is_error,
      result: overrides.result ?? "",
      errors: overrides.errors ?? [],
      duration_ms: 0,
      duration_api_ms: 0,
      num_turns: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: randomUUID(),
      session_id: "test-session",
    };
  }

  function injectSession(agent: ClaudeAcpAgent, messages: any[]) {
    const input = new Pushable<any>();
    async function* messageGenerator() {
      // Wait for the prompt to push its user message so we can replay it
      const iter = input[Symbol.asyncIterator]();
      const { value: userMessage, done } = await iter.next();
      if (!done && userMessage) {
        yield {
          type: "user",
          message: userMessage.message,
          parent_tool_use_id: null,
          uuid: userMessage.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      yield* messages;
    }
    agent.sessions["test-session"] = {
      query: messageGenerator() as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: {
        currentModeId: "default",
        availableModes: [],
      },
      models: {
        currentModelId: "default",
        availableModels: [],
      },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");
  }

  it("should return max_tokens when success result has stop_reason max_tokens", async () => {
    const agent = createMockAgent();
    injectSession(agent, [
      createResultMessage({ subtype: "success", stop_reason: "max_tokens", is_error: false }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    const response = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "test" }],
    });

    expect(response.stopReason).toBe("max_tokens");
  });

  it("should return max_tokens when success result has stop_reason max_tokens and is_error true", async () => {
    const agent = createMockAgent();
    injectSession(agent, [
      createResultMessage({
        subtype: "success",
        stop_reason: "max_tokens",
        is_error: true,
        result: "Token limit reached",
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    const response = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "test" }],
    });

    expect(response.stopReason).toBe("max_tokens");
  });

  it("should return max_tokens when error_during_execution has stop_reason max_tokens", async () => {
    const agent = createMockAgent();
    injectSession(agent, [
      createResultMessage({
        subtype: "error_during_execution",
        stop_reason: "max_tokens",
        is_error: true,
        errors: ["some error"],
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    const response = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "test" }],
    });

    expect(response.stopReason).toBe("max_tokens");
  });

  it("should return end_turn for success with null stop_reason", async () => {
    const agent = createMockAgent();
    injectSession(agent, [
      createResultMessage({ subtype: "success", stop_reason: null, is_error: false }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    const response = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "test" }],
    });

    expect(response.stopReason).toBe("end_turn");
  });

  it("should consume background task results and return the prompt's own result", async () => {
    const agent = createMockAgent();
    const input = new Pushable<any>();

    const backgroundTaskResult = createResultMessage({
      subtype: "success",
      stop_reason: null,
      is_error: false,
    });
    // Background task used some tokens
    backgroundTaskResult.usage.input_tokens = 100;
    backgroundTaskResult.usage.output_tokens = 50;

    const promptResult = createResultMessage({
      subtype: "success",
      stop_reason: null,
      is_error: false,
    });

    async function* messageGenerator() {
      // Background task init + result arrive before our prompt's replay
      yield { type: "system", subtype: "init", session_id: "test-session" };
      yield backgroundTaskResult;

      // Now the prompt's user message replay arrives
      const iter = input[Symbol.asyncIterator]();
      const { value: userMessage } = await iter.next();
      yield {
        type: "user",
        message: userMessage.message,
        parent_tool_use_id: null,
        uuid: userMessage.uuid,
        session_id: "test-session",
        isReplay: true,
      };

      // Then the prompt's own result
      yield promptResult;
      yield { type: "system", subtype: "session_state_changed", state: "idle" };
    }

    agent.sessions["test-session"] = {
      query: messageGenerator() as any,
      input,
      cwd: "/tmp/test",
      sessionFingerprint: JSON.stringify({ cwd: "/tmp/test", mcpServers: [] }),
      cancelled: false,
      modes: {
        currentModeId: "default",
        availableModes: [],
      },
      models: {
        currentModelId: "default",
        availableModels: [],
      },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      abortController: new AbortController(),
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");

    const response = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "test" }],
    });

    expect(response.stopReason).toBe("end_turn");
    // Usage should include both background task and prompt result tokens
    expect(response.usage?.inputTokens).toBe(
      backgroundTaskResult.usage.input_tokens + promptResult.usage.input_tokens,
    );
    expect(response.usage?.outputTokens).toBe(
      backgroundTaskResult.usage.output_tokens + promptResult.usage.output_tokens,
    );
  });

  it("should throw internal error for success with is_error true and no max_tokens", async () => {
    const agent = createMockAgent();
    injectSession(agent, [
      createResultMessage({
        subtype: "success",
        stop_reason: "end_turn",
        is_error: true,
        result: "Something went wrong",
      }),
    ]);

    await expect(
      agent.prompt({
        sessionId: "test-session",
        prompt: [{ type: "text", text: "test" }],
      }),
    ).rejects.toThrow("Internal error");
  });

  it("forwards SDKAssistantMessage.error as structured data on internal errors", async () => {
    const agent = createMockAgent();
    const assistantMessage: SDKAssistantMessage = {
      type: "assistant",
      parent_tool_use_id: null,
      error: "rate_limit",
      uuid: randomUUID(),
      session_id: "test-session",
      message: {
        id: "msg-1",
        type: "message",
        role: "assistant",
        container: null,
        model: "claude-sonnet-4-20250514",
        content: [],
        stop_reason: "stop_sequence",
        stop_sequence: null,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
          server_tool_use: { web_search_requests: 0, web_fetch_requests: 0 },
          service_tier: null,
          cache_creation: {
            ephemeral_1h_input_tokens: 0,
            ephemeral_5m_input_tokens: 0,
          },
        } as any,
      } as any,
    };

    injectSession(agent, [
      assistantMessage,
      createResultMessage({
        subtype: "success",
        stop_reason: "end_turn",
        is_error: true,
        result: "You've hit your limit · resets 8pm",
      }),
    ]);

    const err = await agent
      .prompt({
        sessionId: "test-session",
        prompt: [{ type: "text", text: "test" }],
      })
      .then(
        () => null,
        (e) => e,
      );

    expect(err).not.toBeNull();
    expect((err as { data: unknown }).data).toEqual({ errorKind: "rate_limit" });
  });

  it("omits errorKind data when no SDKAssistantMessage.error was observed", async () => {
    const agent = createMockAgent();
    injectSession(agent, [
      createResultMessage({
        subtype: "success",
        stop_reason: "end_turn",
        is_error: true,
        result: "Something went wrong",
      }),
    ]);

    const err = await agent
      .prompt({
        sessionId: "test-session",
        prompt: [{ type: "text", text: "test" }],
      })
      .then(
        () => null,
        (e) => e,
      );

    expect(err).not.toBeNull();
    expect((err as { data: unknown }).data).toBeUndefined();
  });
});

describe("session/close", () => {
  function createMockAgent() {
    const mockClient = {
      sessionUpdate: async () => {},
    } as unknown as AgentSideConnection;
    return new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
  }

  function injectSession(agent: ClaudeAcpAgent, sessionId: string) {
    function* empty() {}
    const gen = Object.assign(empty(), { interrupt: vi.fn(), close: vi.fn() });
    agent.sessions[sessionId] = {
      query: gen as any,
      input: new Pushable(),
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: {
        currentModeId: "default",
        availableModes: [],
      },
      models: {
        currentModelId: "default",
        availableModels: [],
      },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, sessionId);
    return agent.sessions[sessionId]!;
  }

  it("should close an existing session and remove it", async () => {
    const agent = createMockAgent();
    const session = injectSession(agent, "session-1");

    expect(agent.sessions["session-1"]).toBeDefined();

    const result = await agent.closeSession({ sessionId: "session-1" });

    expect(result).toEqual({});
    expect(agent.sessions["session-1"]).toBeUndefined();
    expect(session.query.interrupt).toHaveBeenCalled();
    expect(session.settingsManager.dispose).toHaveBeenCalled();
  });

  it("should abort the session's abort controller", async () => {
    const agent = createMockAgent();
    const session = injectSession(agent, "session-2");

    expect(session.abortController.signal.aborted).toBe(false);

    await agent.closeSession({ sessionId: "session-2" });

    expect(session.abortController.signal.aborted).toBe(true);
  });

  it("should throw when closing a non-existent session", async () => {
    const agent = createMockAgent();

    await expect(agent.closeSession({ sessionId: "non-existent" })).rejects.toThrow(
      "Session not found",
    );
  });

  it("should not affect other sessions when closing one", async () => {
    const agent = createMockAgent();
    injectSession(agent, "session-a");
    injectSession(agent, "session-b");

    await agent.closeSession({ sessionId: "session-a" });

    expect(agent.sessions["session-a"]).toBeUndefined();
    expect(agent.sessions["session-b"]).toBeDefined();
  });
});

describe("session/delete", () => {
  function createMockAgent() {
    const mockClient = {
      sessionUpdate: async () => {},
    } as unknown as AgentSideConnection;
    return new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
  }

  function injectSession(agent: ClaudeAcpAgent, sessionId: string) {
    function* empty() {}
    const gen = Object.assign(empty(), { interrupt: vi.fn(), close: vi.fn() });
    agent.sessions[sessionId] = {
      query: gen as any,
      input: new Pushable(),
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, sessionId);
    return agent.sessions[sessionId]!;
  }

  beforeEach(() => {
    vi.mocked(deleteSession).mockReset();
    vi.mocked(deleteSession).mockResolvedValue(undefined);
  });

  it("tears down the active session and deletes it from disk", async () => {
    const agent = createMockAgent();
    const session = injectSession(agent, "session-1");

    const result = await agent.deleteSession({ sessionId: "session-1" });

    expect(result).toEqual({});
    expect(agent.sessions["session-1"]).toBeUndefined();
    expect(session.query.interrupt).toHaveBeenCalled();
    expect(session.settingsManager.dispose).toHaveBeenCalled();
    expect(session.abortController.signal.aborted).toBe(true);
    expect(deleteSession).toHaveBeenCalledWith("session-1");
  });

  it("deletes a session from disk that is not currently active", async () => {
    const agent = createMockAgent();

    const result = await agent.deleteSession({ sessionId: "not-active" });

    expect(result).toEqual({});
    expect(deleteSession).toHaveBeenCalledWith("not-active");
  });

  it("propagates errors from the SDK delete call", async () => {
    const agent = createMockAgent();
    vi.mocked(deleteSession).mockRejectedValueOnce(new Error("Session not found on disk"));

    await expect(agent.deleteSession({ sessionId: "missing" })).rejects.toThrow(
      "Session not found on disk",
    );
  });

  it("does not affect other sessions when deleting one", async () => {
    const agent = createMockAgent();
    injectSession(agent, "session-a");
    injectSession(agent, "session-b");

    await agent.deleteSession({ sessionId: "session-a" });

    expect(agent.sessions["session-a"]).toBeUndefined();
    expect(agent.sessions["session-b"]).toBeDefined();
  });
});

describe("getOrCreateSession param change detection", () => {
  function createMockAgent() {
    const mockClient = {
      sessionUpdate: async () => {},
    } as unknown as AgentSideConnection;
    return new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
  }

  function injectSession(
    agent: ClaudeAcpAgent,
    sessionId: string,
    opts: { cwd?: string; mcpServers?: { name: string }[] } = {},
  ) {
    const cwd = opts.cwd ?? "/test";
    const mcpServers = (opts.mcpServers ?? []) as any[];
    function* empty() {}
    const gen = Object.assign(empty(), {
      interrupt: vi.fn(),
      close: vi.fn(),
      supportedCommands: vi.fn().mockResolvedValue([]),
    });
    agent.sessions[sessionId] = {
      query: gen as any,
      input: new Pushable(),
      cancelled: false,
      cwd,
      sessionFingerprint: JSON.stringify({
        cwd,
        mcpServers: [...mcpServers].sort((a: any, b: any) => a.name.localeCompare(b.name)),
      }),
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, sessionId);
    return agent.sessions[sessionId]!;
  }

  it("returns cached session when params are unchanged", async () => {
    const agent = createMockAgent();
    const session = injectSession(agent, "s1", { cwd: "/project" });

    await agent.resumeSession({
      sessionId: "s1",
      cwd: "/project",
      mcpServers: [],
    });

    // Session object should be the exact same reference (not recreated)
    expect(agent.sessions["s1"]).toBe(session);
    expect(session.settingsManager.dispose).not.toHaveBeenCalled();
  });

  it("tears down existing session when cwd changes", async () => {
    const agent = createMockAgent();
    const session = injectSession(agent, "s1", { cwd: "/old" });

    // Mock createSession to avoid spawning a real process.
    // It will throw, but we can catch that — we only need to verify
    // the old session was torn down before createSession was attempted.
    const createSessionSpy = vi
      .spyOn(agent as any, "createSession")
      .mockRejectedValue(new Error("mock"));

    await expect(
      agent.resumeSession({ sessionId: "s1", cwd: "/new", mcpServers: [] }),
    ).rejects.toThrow("mock");

    // Old session should have been fully torn down
    expect(session.settingsManager.dispose).toHaveBeenCalled();
    expect(session.abortController.signal.aborted).toBe(true);
    expect(session.query.interrupt).toHaveBeenCalled();
    expect(agent.sessions["s1"]).toBeUndefined();

    // createSession should have been called with the new cwd
    expect(createSessionSpy).toHaveBeenCalledWith(
      expect.objectContaining({ cwd: "/new" }),
      expect.objectContaining({ resume: "s1" }),
    );
  });

  it("tears down existing session when mcpServers change", async () => {
    const agent = createMockAgent();
    const session = injectSession(agent, "s1", { cwd: "/project" });

    const createSessionSpy = vi
      .spyOn(agent as any, "createSession")
      .mockRejectedValue(new Error("mock"));

    await expect(
      agent.resumeSession({
        sessionId: "s1",
        cwd: "/project",
        mcpServers: [{ name: "new-server", command: "node", args: ["server.js"], env: [] }],
      }),
    ).rejects.toThrow("mock");

    expect(session.settingsManager.dispose).toHaveBeenCalled();
    expect(session.abortController.signal.aborted).toBe(true);
    expect(agent.sessions["s1"]).toBeUndefined();
    expect(createSessionSpy).toHaveBeenCalled();
  });

  it("treats mcpServers in different order as unchanged", async () => {
    const agent = createMockAgent();
    const servers = [
      { name: "b-server", command: "node", args: ["b.js"], env: [] },
      { name: "a-server", command: "node", args: ["a.js"], env: [] },
    ] as const;
    const session = injectSession(agent, "s1", {
      cwd: "/project",
      mcpServers: servers as any,
    });

    // Same servers but reversed order — should NOT trigger teardown
    await agent.resumeSession({
      sessionId: "s1",
      cwd: "/project",
      mcpServers: [...servers].reverse() as any,
    });

    expect(agent.sessions["s1"]).toBe(session);
    expect(session.settingsManager.dispose).not.toHaveBeenCalled();
  });
});

describe("usage_update computation", () => {
  function createAssistantMessage(overrides: {
    model: string;
    usage?: {
      input_tokens: number;
      output_tokens: number;
      cache_read_input_tokens: number;
      cache_creation_input_tokens: number;
    };
  }) {
    return {
      type: "assistant" as const,
      parent_tool_use_id: null,
      uuid: randomUUID(),
      session_id: "test-session",
      message: {
        model: overrides.model,
        content: [{ type: "text", text: "hello" }],
        usage: overrides.usage ?? {
          input_tokens: 100,
          output_tokens: 50,
          cache_read_input_tokens: 20,
          cache_creation_input_tokens: 10,
        },
      },
    };
  }

  function createResultMessageWithModel(overrides: {
    modelUsage: Record<
      string,
      {
        inputTokens: number;
        outputTokens: number;
        cacheReadInputTokens: number;
        cacheCreationInputTokens: number;
        webSearchRequests: number;
        costUSD: number;
        contextWindow: number;
        maxOutputTokens: number;
      }
    >;
  }) {
    return {
      type: "result" as const,
      subtype: "success" as const,
      stop_reason: "end_turn",
      is_error: false,
      result: "",
      errors: [],
      duration_ms: 0,
      duration_api_ms: 0,
      num_turns: 1,
      total_cost_usd: 0.01,
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: overrides.modelUsage,
      permission_denials: [],
      uuid: randomUUID(),
      session_id: "test-session",
    };
  }

  function createStreamEvent(
    eventType: "message_start" | "message_delta",
    payload: Record<string, unknown>,
    parentToolUseId: string | null = null,
  ) {
    return {
      type: "stream_event" as const,
      parent_tool_use_id: parentToolUseId,
      uuid: randomUUID(),
      session_id: "test-session",
      event:
        eventType === "message_start"
          ? { type: "message_start" as const, message: payload }
          : { type: "message_delta" as const, ...payload },
    };
  }

  function createMockAgentWithCapture() {
    const updates: any[] = [];
    const mockClient = {
      sessionUpdate: async (notification: any) => {
        updates.push(notification);
      },
    } as unknown as AgentSideConnection;
    const agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
    return { agent, updates };
  }

  it("computes followup usage from buffered stream snapshots", () => {
    const result = createResultMessageWithModel({
      modelUsage: {
        "claude-opus-4-20250514": {
          inputTokens: 1000,
          outputTokens: 500,
          cacheReadInputTokens: 200,
          cacheCreationInputTokens: 100,
          webSearchRequests: 0,
          costUSD: 0.01,
          contextWindow: 1000000,
          maxOutputTokens: 16384,
        },
      },
    });
    result.usage = {
      input_tokens: 1,
      output_tokens: 1,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    };

    const update = computeFollowupUsageUpdate(
      [
        createStreamEvent("message_start", {
          model: "claude-opus-4-20250514",
          usage: {
            input_tokens: 1000,
            output_tokens: 0,
            cache_read_input_tokens: 200,
            cache_creation_input_tokens: 100,
          },
        }) as any,
        createStreamEvent("message_delta", {
          usage: { output_tokens: 500 },
        }) as any,
      ],
      result as any,
      200000,
    );

    expect(update).toEqual({ used: 1800, size: 1000000 });
  });

  it("falls back to result usage for followups without a buffered usage snapshot", () => {
    const result = createResultMessageWithModel({ modelUsage: {} });
    result.usage = {
      input_tokens: 10,
      output_tokens: 5,
      cache_read_input_tokens: 2,
      cache_creation_input_tokens: 1,
    };

    expect(computeFollowupUsageUpdate([], result as any, 200000)).toEqual({
      used: 18,
      size: 200000,
    });
  });

  it("ignores subagent (non-top-level) messages and falls back to result usage", () => {
    const result = createResultMessageWithModel({ modelUsage: {} });
    result.usage = {
      input_tokens: 30,
      output_tokens: 7,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    };

    // Every buffered message belongs to a subagent (parent_tool_use_id set),
    // so no top-level snapshot is captured and `used` falls back to
    // result.usage rather than counting subagent token spend.
    const update = computeFollowupUsageUpdate(
      [
        createStreamEvent(
          "message_start",
          {
            model: "claude-opus-4-20250514",
            usage: {
              input_tokens: 99999,
              output_tokens: 0,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
          },
          "subagent-tool-1",
        ) as any,
        createStreamEvent(
          "message_delta",
          { usage: { output_tokens: 88888 } },
          "subagent-tool-1",
        ) as any,
      ],
      result as any,
      200000,
    );

    expect(update).toEqual({ used: 37, size: 200000 });
  });

  it("infers context window size when the model has no matching modelUsage key", () => {
    // modelUsage doesn't contain the buffered model, so size resolution skips
    // getMatchingModelUsage and uses inferContextWindowFromModel: a '[1m]'
    // model infers 1,000,000 rather than the fallback.
    const result = createResultMessageWithModel({ modelUsage: {} });
    result.usage = {
      input_tokens: 1,
      output_tokens: 1,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    };

    const update = computeFollowupUsageUpdate(
      [
        createStreamEvent("message_start", {
          model: "claude-sonnet-4-6[1m]",
          usage: {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        }) as any,
      ],
      result as any,
      200000,
    );

    expect(update.used).toBe(150);
    expect(update.size).toBe(1000000);
  });

  function injectSession(agent: ClaudeAcpAgent, messages: any[]) {
    const input = new Pushable<any>();
    async function* messageGenerator() {
      // Wait for the prompt to push its user message so we can replay it
      const iter = input[Symbol.asyncIterator]();
      const { value: userMessage, done } = await iter.next();
      if (!done && userMessage) {
        yield {
          type: "user",
          message: userMessage.message,
          parent_tool_use_id: null,
          uuid: userMessage.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      yield* messages;
    }
    agent.sessions["test-session"] = {
      query: messageGenerator() as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: {
        currentModeId: "default",
        availableModes: [],
      },
      models: {
        currentModelId: "default",
        availableModels: [],
      },
      modelInfos: [],
      settingsManager: {} as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");
  }

  it("used sums all token types as post-turn context occupancy proxy", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createAssistantMessage({
        model: "claude-opus-4-20250514",
        usage: {
          input_tokens: 1000,
          output_tokens: 500,
          cache_read_input_tokens: 200,
          cache_creation_input_tokens: 100,
        },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 1000,
            outputTokens: 500,
            cacheReadInputTokens: 200,
            cacheCreationInputTokens: 100,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    // used = input(1000) + output(500) + cache_read(200) + cache_creation(100) = 1800
    expect(usageUpdate.update.used).toBe(1800);
  });

  it("coerces null input/output tokens so wire `used` is never null", async () => {
    // Synthetic or third-party-backend stream events have been observed
    // emitting input_tokens/output_tokens as null. Without coercion the
    // snapshot leaks NaN into totalTokens(), and JSON.stringify(NaN) === "null"
    // produces a malformed `used: null` that schema-validating ACP clients reject.
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createAssistantMessage({
        model: "claude-opus-4-20250514",
        usage: {
          input_tokens: null,
          output_tokens: null,
          cache_read_input_tokens: 200,
          cache_creation_input_tokens: 100,
        } as unknown as {
          input_tokens: number;
          output_tokens: number;
          cache_read_input_tokens: number;
          cache_creation_input_tokens: number;
        },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 0,
            outputTokens: 0,
            cacheReadInputTokens: 200,
            cacheCreationInputTokens: 100,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates.length).toBeGreaterThan(0);
    for (const u of usageUpdates) {
      expect(u.update.used).not.toBeNull();
      expect(Number.isFinite(u.update.used)).toBe(true);
      // Round-trip through JSON to catch the NaN -> "null" serialization bug.
      const wire = JSON.parse(JSON.stringify(u.update));
      expect(wire.used).not.toBeNull();
      expect(typeof wire.used).toBe("number");
    }
  });

  it("stream_event message_start emits usage_update before result", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent("message_start", {
        model: "claude-opus-4-20250514",
        usage: {
          input_tokens: 1000,
          output_tokens: 500,
          cache_read_input_tokens: 200,
          cache_creation_input_tokens: 100,
        },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 1000,
            outputTokens: 500,
            cacheReadInputTokens: 200,
            cacheCreationInputTokens: 100,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates).toHaveLength(2);
    expect(usageUpdates[0].update.used).toBe(1800);
    // First prompt of a session has no prior result to learn the window from,
    // so the mid-stream update falls back to the default context window.
    expect(usageUpdates[0].update.size).toBe(200000);
    expect(usageUpdates[0].update.cost).toBeUndefined();
    expect(usageUpdates[1].update.used).toBe(1800);
    expect(usageUpdates[1].update.size).toBe(1000000);
    expect(usageUpdates[1].update.cost).toBeDefined();
  });

  it("stream_event message_delta patches previous snapshot", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent("message_start", {
        model: "claude-opus-4-20250514",
        usage: {
          input_tokens: 1000,
          output_tokens: 0,
          cache_read_input_tokens: 200,
          cache_creation_input_tokens: 100,
        },
      }),
      createStreamEvent("message_delta", {
        usage: { output_tokens: 500 },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 1000,
            outputTokens: 500,
            cacheReadInputTokens: 200,
            cacheCreationInputTokens: 100,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates).toHaveLength(3);
    expect(usageUpdates[0].update.used).toBe(1300);
    expect(usageUpdates[0].update.cost).toBeUndefined();
    expect(usageUpdates[1].update.used).toBe(1800);
    expect(usageUpdates[1].update.cost).toBeUndefined();
    expect(usageUpdates[2].update.used).toBe(1800);
    expect(usageUpdates[2].update.cost).toBeDefined();
  });

  it("mid-stream size is inferred from a 1M model name before the first result", async () => {
    // On the very first prompt there is no learned context window yet, so the
    // mid-stream update would otherwise fall back to 200k. A "-1m" suffix in
    // the SDK model ID is enough signal to emit 1_000_000 up front.
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent("message_start", {
        model: "claude-opus-4-6-1m",
        usage: {
          input_tokens: 2000,
          output_tokens: 1000,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-6-1m": {
            inputTokens: 2000,
            outputTokens: 1000,
            cacheReadInputTokens: 0,
            cacheCreationInputTokens: 0,
            webSearchRequests: 0,
            costUSD: 0.02,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates).toHaveLength(2);
    expect(usageUpdates[0].update.size).toBe(1000000);
    expect(usageUpdates[1].update.size).toBe(1000000);
  });

  it("duplicate stream_event totals do not re-emit usage_update", async () => {
    // A message_delta whose cumulative totals match the prior snapshot should
    // not trigger a duplicate usage_update — only the result adds cost on top.
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent("message_start", {
        model: "claude-opus-4-20250514",
        usage: {
          input_tokens: 1000,
          output_tokens: 500,
          cache_read_input_tokens: 200,
          cache_creation_input_tokens: 100,
        },
      }),
      createStreamEvent("message_delta", {
        usage: { output_tokens: 500 },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 1000,
            outputTokens: 500,
            cacheReadInputTokens: 200,
            cacheCreationInputTokens: 100,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates).toHaveLength(2);
    expect(usageUpdates[0].update.used).toBe(1800);
    expect(usageUpdates[0].update.cost).toBeUndefined();
    expect(usageUpdates[1].update.used).toBe(1800);
    expect(usageUpdates[1].update.cost).toBeDefined();
  });

  it("mid-stream size uses the session's learned context window", async () => {
    // Session state persists the model's context window across prompts, so a
    // mid-stream update in a later prompt reports the real size immediately
    // instead of snapping back to the 200k default before the result arrives.
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent("message_start", {
        model: "claude-opus-4-20250514",
        usage: {
          input_tokens: 2000,
          output_tokens: 1000,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 2000,
            outputTokens: 1000,
            cacheReadInputTokens: 0,
            cacheCreationInputTokens: 0,
            webSearchRequests: 0,
            costUSD: 0.02,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);
    // Simulate a prior prompt having learned the 1M window for this model.
    agent.sessions["test-session"].contextWindowSize = 1000000;

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates).toHaveLength(2);
    expect(usageUpdates[0].update.size).toBe(1000000);
    expect(usageUpdates[1].update.size).toBe(1000000);
  });

  it("switching to a 1M model seeds the context window from the heuristic", async () => {
    // The heuristic runs at config-change time so mid-stream updates in the
    // next prompt already report 1M — without waiting for message_start or
    // the next `result` to correct us.
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent("message_start", {
        model: "claude-opus-4-6-1m",
        usage: {
          input_tokens: 2000,
          output_tokens: 1000,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-6-1m": {
            inputTokens: 2000,
            outputTokens: 1000,
            cacheReadInputTokens: 0,
            cacheCreationInputTokens: 0,
            webSearchRequests: 0,
            costUSD: 0.02,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);
    const session = agent.sessions["test-session"];
    expect(session.contextWindowSize).toBe(200000);

    await (agent as any).applyConfigOptionValue(
      "test-session",
      session,
      "model",
      "claude-opus-4-6-1m",
    );
    expect(session.contextWindowSize).toBe(1000000);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates).toHaveLength(2);
    expect(usageUpdates[0].update.size).toBe(1000000);
    expect(usageUpdates[1].update.size).toBe(1000000);
  });

  it("result with no matching modelUsage preserves the learned window", async () => {
    // A turn whose `result.modelUsage` doesn't contain the current top-level
    // model (e.g. no top-level assistant message, or only a subagent ran) must
    // not clobber the window learned on a prior turn — otherwise the next
    // prompt's mid-stream updates regress to the 200k default.
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createResultMessageWithModel({
        modelUsage: {
          "claude-haiku-4-5-20251001": {
            inputTokens: 100,
            outputTokens: 50,
            cacheReadInputTokens: 0,
            cacheCreationInputTokens: 0,
            webSearchRequests: 0,
            costUSD: 0.001,
            contextWindow: 200000,
            maxOutputTokens: 8192,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);
    const session = agent.sessions["test-session"];
    session.contextWindowSize = 1000000;

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    expect(session.contextWindowSize).toBe(1000000);
    // The emit itself falls back to session.contextWindowSize, which is
    // unchanged from the learned value.
    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    // No lastAssistantTotalUsage was set (no top-level assistant / stream
    // event), so the result branch skips its emit entirely.
    expect(usageUpdates).toHaveLength(0);
  });

  it("switching the session's model invalidates the learned context window", async () => {
    // When the user switches models mid-session, the window learned for the
    // previous model would otherwise persist into the next prompt's first
    // mid-stream update. applyConfigOptionValue should reset it so the next
    // turn's first update falls back to the heuristic (here: 200k default).
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent("message_start", {
        model: "claude-sonnet-4-6",
        usage: {
          input_tokens: 1000,
          output_tokens: 500,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-sonnet-4-6": {
            inputTokens: 1000,
            outputTokens: 500,
            cacheReadInputTokens: 0,
            cacheCreationInputTokens: 0,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 200000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);
    const session = agent.sessions["test-session"];
    session.contextWindowSize = 1000000;
    session.models = { ...session.models, currentModelId: "claude-opus-4-6-1m" };

    // User flips the selector to a 200k model.
    await (agent as any).applyConfigOptionValue(
      "test-session",
      session,
      "model",
      "claude-sonnet-4-6",
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates).toHaveLength(2);
    expect(usageUpdates[0].update.size).toBe(200000);
    expect(usageUpdates[1].update.size).toBe(200000);
  });

  it("non-usage stream events do not re-emit usage_update", async () => {
    // content_block_* and message_stop carry no usage fields; they must not
    // trigger duplicate emits between the real message_start / message_delta
    // / result updates.
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent("message_start", {
        model: "claude-opus-4-20250514",
        usage: {
          input_tokens: 1000,
          output_tokens: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      }),
      {
        type: "stream_event" as const,
        parent_tool_use_id: null,
        uuid: randomUUID(),
        session_id: "test-session",
        event: { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      },
      {
        type: "stream_event" as const,
        parent_tool_use_id: null,
        uuid: randomUUID(),
        session_id: "test-session",
        event: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } },
      },
      {
        type: "stream_event" as const,
        parent_tool_use_id: null,
        uuid: randomUUID(),
        session_id: "test-session",
        event: { type: "content_block_stop", index: 0 },
      },
      createStreamEvent("message_delta", {
        usage: { output_tokens: 200 },
      }),
      {
        type: "stream_event" as const,
        parent_tool_use_id: null,
        uuid: randomUUID(),
        session_id: "test-session",
        event: { type: "message_stop" },
      },
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 1000,
            outputTokens: 200,
            cacheReadInputTokens: 0,
            cacheCreationInputTokens: 0,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    // Exactly three: message_start (1000), message_delta (1200), result (1200 + cost).
    expect(usageUpdates).toHaveLength(3);
    expect(usageUpdates[0].update.used).toBe(1000);
    expect(usageUpdates[1].update.used).toBe(1200);
    expect(usageUpdates[2].update.used).toBe(1200);
    expect(usageUpdates[2].update.cost).toBeDefined();
  });

  it("subagent stream_event does not emit usage_update", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createStreamEvent(
        "message_start",
        {
          model: "claude-haiku-4-5-20251001",
          usage: {
            input_tokens: 500,
            output_tokens: 100,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
        "tool_use_123",
      ),
      createResultMessageWithModel({
        modelUsage: {
          "claude-haiku-4-5-20251001": {
            inputTokens: 500,
            outputTokens: 100,
            cacheReadInputTokens: 0,
            cacheCreationInputTokens: 0,
            webSearchRequests: 0,
            costUSD: 0.001,
            contextWindow: 200000,
            maxOutputTokens: 8192,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdates = updates.filter((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdates).toHaveLength(0);
  });

  it("size reflects the current model's context window, not min across all", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createAssistantMessage({ model: "claude-opus-4-20250514" }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 100,
            outputTokens: 50,
            cacheReadInputTokens: 20,
            cacheCreationInputTokens: 10,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
          "claude-sonnet-4-20250514": {
            inputTokens: 50,
            outputTokens: 25,
            cacheReadInputTokens: 10,
            cacheCreationInputTokens: 5,
            webSearchRequests: 0,
            costUSD: 0.005,
            contextWindow: 200000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    // size should be 1000000 (Opus), not 200000 (min of both)
    expect(usageUpdate.update.size).toBe(1000000);
  });

  it("after model switch, size updates to the new model's window", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Simulate: assistant on Sonnet with both models in modelUsage
    injectSession(agent, [
      createAssistantMessage({ model: "claude-sonnet-4-20250514" }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 100,
            outputTokens: 50,
            cacheReadInputTokens: 20,
            cacheCreationInputTokens: 10,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
          "claude-sonnet-4-20250514": {
            inputTokens: 50,
            outputTokens: 25,
            cacheReadInputTokens: 10,
            cacheCreationInputTokens: 5,
            webSearchRequests: 0,
            costUSD: 0.005,
            contextWindow: 200000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    // size should be 200000 (Sonnet - the current model)
    expect(usageUpdate.update.size).toBe(200000);
  });

  it("after switching back to original model, size returns to original window", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Last assistant message is Opus again
    injectSession(agent, [
      createAssistantMessage({ model: "claude-sonnet-4-20250514" }),
      createAssistantMessage({ model: "claude-opus-4-20250514" }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 200,
            outputTokens: 100,
            cacheReadInputTokens: 40,
            cacheCreationInputTokens: 20,
            webSearchRequests: 0,
            costUSD: 0.02,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
          "claude-sonnet-4-20250514": {
            inputTokens: 50,
            outputTokens: 25,
            cacheReadInputTokens: 10,
            cacheCreationInputTokens: 5,
            webSearchRequests: 0,
            costUSD: 0.005,
            contextWindow: 200000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    // size should be 1000000 (Opus - switched back)
    expect(usageUpdate.update.size).toBe(1000000);
  });

  it("subagent assistant messages do not affect size (top-level model is used)", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Top-level assistant on Opus, then subagent on Haiku (parent_tool_use_id set)
    injectSession(agent, [
      createAssistantMessage({ model: "claude-opus-4-20250514" }),
      {
        type: "assistant" as const,
        parent_tool_use_id: "tool_use_123",
        uuid: randomUUID(),
        session_id: "test-session",
        message: {
          model: "claude-haiku-4-5-20251001",
          content: [{ type: "text", text: "subagent response" }],
          usage: {
            input_tokens: 50,
            output_tokens: 25,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 100,
            outputTokens: 50,
            cacheReadInputTokens: 20,
            cacheCreationInputTokens: 10,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
          "claude-haiku-4-5-20251001": {
            inputTokens: 50,
            outputTokens: 25,
            cacheReadInputTokens: 0,
            cacheCreationInputTokens: 0,
            webSearchRequests: 0,
            costUSD: 0.001,
            contextWindow: 200000,
            maxOutputTokens: 8192,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    // size should be 1000000 (Opus - the top-level model), NOT 200000 (Haiku subagent)
    expect(usageUpdate.update.size).toBe(1000000);
  });

  it("prefix-matches when assistant model has date suffix but modelUsage key does not", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // The API response has the full versioned model ID on assistant messages,
    // but the SDK's streaming path may key modelUsage by the shorter alias.
    injectSession(agent, [
      createAssistantMessage({ model: "claude-opus-4-6-20250514" }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-6": {
            inputTokens: 100,
            outputTokens: 50,
            cacheReadInputTokens: 20,
            cacheCreationInputTokens: 10,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    // Should match via prefix: "claude-opus-4-6-20250514".startsWith("claude-opus-4-6")
    expect(usageUpdate.update.size).toBe(1000000);
  });

  it("prefix-matches when modelUsage key has date suffix but assistant model does not", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createAssistantMessage({ model: "claude-opus-4-6" }),
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-6-20250514": {
            inputTokens: 100,
            outputTokens: 50,
            cacheReadInputTokens: 20,
            cacheCreationInputTokens: 10,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    expect(usageUpdate.update.size).toBe(1000000);
  });

  it("synthetic assistant messages do not override lastAssistantModel", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Real assistant on Opus, then a synthetic message (e.g. from /compact)
    injectSession(agent, [
      createAssistantMessage({ model: "claude-opus-4-20250514" }),
      {
        type: "assistant" as const,
        parent_tool_use_id: null,
        uuid: randomUUID(),
        session_id: "test-session",
        message: {
          model: "<synthetic>",
          content: [{ type: "text", text: "compacted" }],
          usage: {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
        },
      },
      createResultMessageWithModel({
        modelUsage: {
          "claude-opus-4-20250514": {
            inputTokens: 100,
            outputTokens: 50,
            cacheReadInputTokens: 20,
            cacheCreationInputTokens: 10,
            webSearchRequests: 0,
            costUSD: 0.01,
            contextWindow: 1000000,
            maxOutputTokens: 16384,
          },
        },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    // size should be 1000000 (Opus), not 200000 (the fallback if <synthetic> overrode the model)
    expect(usageUpdate.update.size).toBe(1000000);
  });

  it("compact_boundary uses authoritative getContextUsage for used, keeps session window for size", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      { type: "system", subtype: "compact_boundary", session_id: "test-session" },
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);
    const session = agent.sessions["test-session"];
    // A 1M window learned earlier (e.g. from modelUsage) must survive compaction
    // — getContextUsage's window field under-reports it, so we don't use it.
    session.contextWindowSize = 1000000;
    (session.query as any).getContextUsage = vi
      .fn()
      .mockResolvedValue({ totalTokens: 12345, rawMaxTokens: 200000 });

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    expect(usageUpdate.update.used).toBe(12345);
    // size stays at the session's learned window, NOT getContextUsage's value.
    expect(usageUpdate.update.size).toBe(1000000);
    expect(session.contextWindowSize).toBe(1000000);
  });

  it("compact_boundary falls back to used:0 when getContextUsage fails", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      { type: "system", subtype: "compact_boundary", session_id: "test-session" },
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);
    const session = agent.sessions["test-session"];
    session.contextWindowSize = 200000;
    (session.query as any).getContextUsage = vi.fn().mockRejectedValue(new Error("boom"));

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    expect(usageUpdate.update.used).toBe(0);
    expect(usageUpdate.update.size).toBe(200000);
    expect(session.contextWindowSize).toBe(200000);
  });
});

describe("assembled assistant text fallback", () => {
  const ZERO_USAGE = {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };

  function createMockAgentWithCapture() {
    const updates: any[] = [];
    const mockClient = {
      sessionUpdate: async (notification: any) => {
        updates.push(notification);
      },
    } as unknown as AgentSideConnection;
    const agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
    return { agent, updates };
  }

  function messageStart(apiId: string) {
    return {
      type: "stream_event" as const,
      parent_tool_use_id: null,
      uuid: randomUUID(),
      session_id: "test-session",
      event: {
        type: "message_start" as const,
        message: { id: apiId, model: "claude-sonnet-4-20250514", usage: ZERO_USAGE },
      },
    };
  }

  function textDelta(text: string) {
    return {
      type: "stream_event" as const,
      parent_tool_use_id: null,
      uuid: randomUUID(),
      session_id: "test-session",
      event: {
        type: "content_block_delta" as const,
        index: 0,
        delta: { type: "text_delta" as const, text },
      },
    };
  }

  function assistantMessage(apiId: string, content: any[], parentToolUseId: string | null = null) {
    return {
      type: "assistant" as const,
      parent_tool_use_id: parentToolUseId,
      uuid: randomUUID(),
      session_id: "test-session",
      message: {
        id: apiId,
        role: "assistant" as const,
        model: "claude-sonnet-4-20250514",
        content,
        usage: ZERO_USAGE,
      },
    };
  }

  function result() {
    return {
      type: "result" as const,
      subtype: "success" as const,
      stop_reason: "end_turn",
      is_error: false,
      result: "",
      errors: [],
      duration_ms: 0,
      duration_api_ms: 0,
      num_turns: 1,
      total_cost_usd: 0,
      usage: ZERO_USAGE,
      modelUsage: {},
      permission_denials: [],
      uuid: randomUUID(),
      session_id: "test-session",
    };
  }

  const idle = { type: "system", subtype: "session_state_changed", state: "idle" };

  function injectSession(agent: ClaudeAcpAgent, messages: any[]) {
    const input = new Pushable<any>();
    async function* messageGenerator() {
      const iter = input[Symbol.asyncIterator]();
      const { value: userMessage, done } = await iter.next();
      if (!done && userMessage) {
        yield {
          type: "user",
          message: userMessage.message,
          parent_tool_use_id: null,
          uuid: userMessage.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      yield* messages;
    }
    agent.sessions["test-session"] = {
      query: messageGenerator() as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");
  }

  function messageChunkTexts(updates: any[]): string[] {
    return updates
      .filter((u) => u.update?.sessionUpdate === "agent_message_chunk")
      .map((u) => u.update.content.text);
  }

  function thoughtChunkTexts(updates: any[]): string[] {
    return updates
      .filter((u) => u.update?.sessionUpdate === "agent_thought_chunk")
      .map((u) => u.update.content.text);
  }

  it("emits the assembled text when no content_block_delta was streamed", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Gateway delivers a fully assembled message with no preceding deltas.
    injectSession(agent, [
      assistantMessage("msg-no-stream", [{ type: "text", text: "the final answer" }]),
      result(),
      idle,
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    expect(messageChunkTexts(updates)).toEqual(["the final answer"]);
  });

  it("does not re-emit text already streamed via content_block_delta", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Normal streaming: deltas arrive, then the consolidated message repeats them.
    injectSession(agent, [
      messageStart("msg-streamed"),
      textDelta("hello "),
      textDelta("world"),
      assistantMessage("msg-streamed", [{ type: "text", text: "hello world" }]),
      result(),
      idle,
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    // Only the two streamed deltas — the assembled block is filtered out.
    expect(messageChunkTexts(updates)).toEqual(["hello ", "world"]);
  });

  it("dedupes per block type: streamed text is dropped but an un-streamed thinking block in the same message is forwarded", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Gateway streams the text live but delivers the thinking block only in the
    // assembled message (no thinking_delta). The dedupe must be per-type so the
    // thinking survives. This also makes the test non-vacuous: if the fallback
    // were removed (text/thinking always dropped) the thought chunk disappears.
    injectSession(agent, [
      messageStart("msg-mixed"),
      textDelta("streamed text"),
      assistantMessage("msg-mixed", [
        { type: "text", text: "streamed text" },
        { type: "thinking", thinking: "private reasoning" },
      ]),
      result(),
      idle,
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    // Streamed text appears once (delta only — assembled copy deduped).
    expect(messageChunkTexts(updates)).toEqual(["streamed text"]);
    // The un-streamed thinking block is forwarded despite text having streamed.
    expect(thoughtChunkTexts(updates)).toEqual(["private reasoning"]);
  });

  it("does not leak subagent assistant text into the top-level feed", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Subagent assistant messages (parent_tool_use_id !== null) are never
    // streamed live; their text/thinking is internal to the tool call and must
    // stay filtered out, not surface as a fallback chunk.
    injectSession(agent, [
      assistantMessage(
        "msg-subagent",
        [{ type: "text", text: "subagent internal prose" }],
        "tool_use_1",
      ),
      result(),
      idle,
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    expect(messageChunkTexts(updates)).toEqual([]);
    expect(thoughtChunkTexts(updates)).toEqual([]);
  });

  it("forwards distinct blocks that a gateway splits across same-id messages", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // Observed with OpenAI-compatible gateways: one response id split into an
    // empty thinking block, then the real text — both with no deltas.
    injectSession(agent, [
      assistantMessage("msg-split", [{ type: "thinking", thinking: "" }]),
      assistantMessage("msg-split", [{ type: "text", text: "the real answer" }]),
      result(),
      idle,
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    // The text survives even though an earlier same-id message already triggered
    // the fallback for a different (thinking) block.
    expect(messageChunkTexts(updates)).toEqual(["the real answer"]);
    // The empty thinking block carries nothing and must not produce a stray
    // empty thought chunk.
    expect(thoughtChunkTexts(updates)).toEqual([]);
  });

  it("re-forwards a block a gateway re-delivers (no content-keyed dedupe)", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    // The fallback intentionally keys only on whether the id streamed live, not
    // on block content — so a gateway re-delivering the same assembled block
    // emits it twice. This is the accepted, cosmetic tradeoff for not caching
    // every fallback block's full text; see `streamedTextMessageIds`.
    injectSession(agent, [
      assistantMessage("msg-dup", [{ type: "text", text: "answer" }]),
      assistantMessage("msg-dup", [{ type: "text", text: "answer" }]),
      result(),
      idle,
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    expect(messageChunkTexts(updates)).toEqual(["answer", "answer"]);
  });
});

describe("emitRawSDKMessages", () => {
  function createMockAgentWithExtNotification() {
    const updates: any[] = [];
    const extNotifications: { method: string; params: any }[] = [];
    const mockClient = {
      sessionUpdate: async (notification: any) => {
        updates.push(notification);
      },
      extNotification: async (method: string, params: any) => {
        extNotifications.push({ method, params });
      },
    } as unknown as AgentSideConnection;
    const agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
    return { agent, updates, extNotifications };
  }

  function injectSession(
    agent: ClaudeAcpAgent,
    messages: any[],
    emitRawSDKMessages: boolean | SDKMessageFilter[],
  ) {
    const input = new Pushable<any>();
    async function* messageGenerator() {
      const iter = input[Symbol.asyncIterator]();
      const { value: userMessage, done } = await iter.next();
      if (!done && userMessage) {
        yield {
          type: "user",
          message: userMessage.message,
          parent_tool_use_id: null,
          uuid: userMessage.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      yield* messages;
    }
    agent.sessions["test-session"] = {
      query: messageGenerator() as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");
  }

  function createResultMessage() {
    return {
      type: "result" as const,
      subtype: "success" as const,
      is_error: false,
      result: "",
      errors: [],
      stop_reason: "end_turn" as const,
      cost_usd: 0,
      duration_ms: 0,
      duration_api_ms: 0,
      num_turns: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: randomUUID(),
      session_id: "test-session",
    };
  }

  it("emits all raw messages when set to true", async () => {
    const { agent, extNotifications } = createMockAgentWithExtNotification();
    const systemMsg = {
      type: "system",
      subtype: "status",
      status: "compacting",
      session_id: "test-session",
    };
    injectSession(
      agent,
      [
        systemMsg,
        createResultMessage(),
        { type: "system", subtype: "session_state_changed", state: "idle" },
      ],
      true,
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    // Should have emitted extNotifications for all messages (user replay + system + result + session_state_changed)
    expect(extNotifications.length).toBeGreaterThanOrEqual(3);
    expect(extNotifications.every((n) => n.method === "_claude/sdkMessage")).toBe(true);
  });

  it("does not emit when set to false", async () => {
    const { agent, extNotifications } = createMockAgentWithExtNotification();
    injectSession(
      agent,
      [
        { type: "system", subtype: "status", status: "compacting", session_id: "test-session" },
        createResultMessage(),
        { type: "system", subtype: "session_state_changed", state: "idle" },
      ],
      false,
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    expect(extNotifications).toHaveLength(0);
  });

  it("emits only messages matching a filter array", async () => {
    const { agent, extNotifications } = createMockAgentWithExtNotification();
    injectSession(
      agent,
      [
        { type: "system", subtype: "compact_boundary", session_id: "test-session" },
        { type: "system", subtype: "status", status: "compacting", session_id: "test-session" },
        createResultMessage(),
        { type: "system", subtype: "session_state_changed", state: "idle" },
      ],
      [{ type: "system", subtype: "compact_boundary" }],
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    // Only the compact_boundary message should have been emitted
    const sdkMessages = extNotifications.filter((n) => n.method === "_claude/sdkMessage");
    expect(sdkMessages).toHaveLength(1);
    expect(sdkMessages[0].params.sessionId).toBe("test-session");
    expect(sdkMessages[0].params.message.type).toBe("system");
    expect(sdkMessages[0].params.message.subtype).toBe("compact_boundary");
  });

  it("filter without subtype matches all messages of that type", async () => {
    const { agent, extNotifications } = createMockAgentWithExtNotification();
    injectSession(
      agent,
      [
        { type: "system", subtype: "compact_boundary", session_id: "test-session" },
        { type: "system", subtype: "status", status: "compacting", session_id: "test-session" },
        createResultMessage(),
        { type: "system", subtype: "session_state_changed", state: "idle" },
      ],
      [{ type: "system" }],
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const sdkMessages = extNotifications.filter((n) => n.method === "_claude/sdkMessage");
    // All system messages should match (compact_boundary + status + session_state_changed)
    const systemMessages = sdkMessages.filter((n) => n.params.message.type === "system");
    expect(systemMessages).toHaveLength(3);
  });

  it("supports multiple filters", async () => {
    const { agent, extNotifications } = createMockAgentWithExtNotification();
    injectSession(
      agent,
      [
        { type: "system", subtype: "compact_boundary", session_id: "test-session" },
        { type: "system", subtype: "status", status: "compacting", session_id: "test-session" },
        createResultMessage(),
        { type: "system", subtype: "session_state_changed", state: "idle" },
      ],
      [{ type: "system", subtype: "compact_boundary" }, { type: "result" }],
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const sdkMessages = extNotifications.filter((n) => n.method === "_claude/sdkMessage");
    expect(sdkMessages).toHaveLength(2);
    expect(sdkMessages[0].params.message.type).toBe("system");
    expect(sdkMessages[0].params.message.subtype).toBe("compact_boundary");
    expect(sdkMessages[1].params.message.type).toBe("result");
  });

  it("filter by origin kind only emits matching results", async () => {
    const { agent, extNotifications } = createMockAgentWithExtNotification();
    injectSession(
      agent,
      [
        { ...createResultMessage(), origin: { kind: "channel", server: "acp" } },
        { ...createResultMessage(), origin: { kind: "task-notification" } },
        { type: "system", subtype: "session_state_changed", state: "idle" },
      ],
      [{ type: "result", origin: "task-notification" }],
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const sdkMessages = extNotifications.filter((n) => n.method === "_claude/sdkMessage");
    expect(sdkMessages).toHaveLength(1);
    expect(sdkMessages[0].params.message.origin.kind).toBe("task-notification");
  });

  it("filter without origin matches results regardless of origin", async () => {
    const { agent, extNotifications } = createMockAgentWithExtNotification();
    injectSession(
      agent,
      [
        { ...createResultMessage(), origin: { kind: "channel", server: "acp" } },
        { ...createResultMessage(), origin: { kind: "task-notification" } },
        { type: "system", subtype: "session_state_changed", state: "idle" },
      ],
      [{ type: "result" }],
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const sdkMessages = extNotifications.filter((n) => n.method === "_claude/sdkMessage");
    expect(sdkMessages).toHaveLength(2);
  });
});

describe("result origin handling", () => {
  function createMockAgentWithCapture() {
    const updates: any[] = [];
    const mockClient = {
      sessionUpdate: async (notification: any) => {
        updates.push(notification);
      },
    } as unknown as AgentSideConnection;
    const agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
    return { agent, updates };
  }

  function injectSession(agent: ClaudeAcpAgent, messages: any[]) {
    const input = new Pushable<any>();
    async function* messageGenerator() {
      const iter = input[Symbol.asyncIterator]();
      const { value: userMessage, done } = await iter.next();
      if (!done && userMessage) {
        yield {
          type: "user",
          message: userMessage.message,
          parent_tool_use_id: null,
          uuid: userMessage.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      yield* messages;
    }
    agent.sessions["test-session"] = {
      query: messageGenerator() as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");
  }

  function createAssistantMessage() {
    return {
      type: "assistant" as const,
      parent_tool_use_id: null,
      uuid: randomUUID(),
      session_id: "test-session",
      message: {
        model: "claude-sonnet-4-6",
        content: [{ type: "text", text: "hello" }],
        usage: {
          input_tokens: 10,
          output_tokens: 5,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
    };
  }

  function createResult(overrides: Record<string, unknown> = {}) {
    return {
      type: "result" as const,
      subtype: "success" as const,
      stop_reason: "end_turn",
      is_error: false,
      result: "",
      errors: [],
      duration_ms: 0,
      duration_api_ms: 0,
      num_turns: 1,
      total_cost_usd: 0.01,
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: randomUUID(),
      session_id: "test-session",
      ...overrides,
    };
  }

  it("forwards origin in usage_update _meta", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createAssistantMessage(),
      createResult({ origin: { kind: "channel", server: "acp" } }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    expect(usageUpdate.update._meta).toEqual({
      "_claude/origin": { kind: "channel", server: "acp" },
    });
  });

  it("omits _meta when origin is absent", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      createAssistantMessage(),
      createResult(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "test" }] });

    const usageUpdate = updates.find((u: any) => u.update?.sessionUpdate === "usage_update");
    expect(usageUpdate).toBeDefined();
    expect(usageUpdate.update._meta).toBeUndefined();
  });

  it("task-notification result with max_tokens does not override the user-turn stopReason", async () => {
    const { agent } = createMockAgentWithCapture();
    injectSession(agent, [
      createAssistantMessage(),
      // User-turn result completes normally
      createResult({ origin: { kind: "channel", server: "acp" } }),
      // Task-notification followup hits max_tokens — must not bleed into the user's stopReason
      createResult({
        stop_reason: "max_tokens",
        origin: { kind: "task-notification" },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    const response = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "test" }],
    });

    expect(response.stopReason).toBe("end_turn");
  });

  it("user-prompted result with max_tokens still sets stopReason", async () => {
    const { agent } = createMockAgentWithCapture();
    injectSession(agent, [
      createAssistantMessage(),
      createResult({
        stop_reason: "max_tokens",
        origin: { kind: "channel", server: "acp" },
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    const response = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "test" }],
    });

    expect(response.stopReason).toBe("max_tokens");
  });
});

describe("memory_recall handling", () => {
  function createMockAgentWithCapture() {
    const updates: any[] = [];
    const mockClient = {
      sessionUpdate: async (notification: any) => {
        updates.push(notification);
      },
    } as unknown as AgentSideConnection;
    const agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
    return { agent, updates };
  }

  function injectSession(agent: ClaudeAcpAgent, messages: any[]) {
    const input = new Pushable<any>();
    async function* messageGenerator() {
      const iter = input[Symbol.asyncIterator]();
      const { value: userMessage, done } = await iter.next();
      if (!done && userMessage) {
        yield {
          type: "user",
          message: userMessage.message,
          parent_tool_use_id: null,
          uuid: userMessage.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      yield* messages;
    }
    agent.sessions["test-session"] = {
      query: messageGenerator() as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");
  }

  function createResult() {
    return {
      type: "result" as const,
      subtype: "success" as const,
      stop_reason: "end_turn",
      is_error: false,
      result: "",
      errors: [],
      duration_ms: 0,
      duration_api_ms: 0,
      num_turns: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: randomUUID(),
      session_id: "test-session",
    };
  }

  it("emits a synthetic tool_call for select mode with one location per memory", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    const recallUuid = randomUUID();
    injectSession(agent, [
      {
        type: "system",
        subtype: "memory_recall",
        mode: "select",
        memories: [
          { path: "/Users/test/.claude/memory/user_role.md", scope: "personal" },
          { path: "/Users/test/.claude/memory/feedback_testing.md", scope: "personal" },
          { path: "/Users/test/.claude/team/conventions.md", scope: "team" },
        ],
        uuid: recallUuid,
        session_id: "test-session",
      },
      createResult(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    const toolCall = updates.find((u: any) => u.update?.sessionUpdate === "tool_call");
    expect(toolCall).toBeDefined();
    expect(toolCall.update).toMatchObject({
      sessionUpdate: "tool_call",
      toolCallId: recallUuid,
      title: "Recalled 3 memories",
      kind: "read",
      status: "completed",
      locations: [
        { path: "/Users/test/.claude/memory/user_role.md" },
        { path: "/Users/test/.claude/memory/feedback_testing.md" },
        { path: "/Users/test/.claude/team/conventions.md" },
      ],
      _meta: {
        claudeCode: { toolName: "memory_recall", toolResponse: { mode: "select" } },
      },
    });
    expect(toolCall.update.content).toBeUndefined();
  });

  it("uses singular 'memory' in title when exactly one entry", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      {
        type: "system",
        subtype: "memory_recall",
        mode: "select",
        memories: [{ path: "/Users/test/.claude/memory/user_role.md", scope: "personal" }],
        uuid: randomUUID(),
        session_id: "test-session",
      },
      createResult(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    const toolCall = updates.find((u: any) => u.update?.sessionUpdate === "tool_call");
    expect(toolCall.update.title).toBe("Recalled 1 memory");
  });

  it("emits synthesis content and no locations for synthesize mode", async () => {
    const { agent, updates } = createMockAgentWithCapture();
    injectSession(agent, [
      {
        type: "system",
        subtype: "memory_recall",
        mode: "synthesize",
        memories: [
          {
            path: "<synthesis:/Users/test/.claude/memory>",
            scope: "personal",
            content: "The user prefers terse responses and writes Go.",
          },
        ],
        uuid: randomUUID(),
        session_id: "test-session",
      },
      createResult(),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "hi" }] });

    const toolCall = updates.find((u: any) => u.update?.sessionUpdate === "tool_call");
    expect(toolCall).toBeDefined();
    expect(toolCall.update.title).toBe("Recalled synthesized memory");
    expect(toolCall.update.locations).toBeUndefined();
    expect(toolCall.update.content).toEqual([
      {
        type: "content",
        content: { type: "text", text: "The user prefers terse responses and writes Go." },
      },
    ]);
    expect(toolCall.update._meta.claudeCode.toolResponse).toEqual({ mode: "synthesize" });
  });
});

describe("post-error recovery", () => {
  function createMockAgent() {
    const mockClient = {
      sessionUpdate: async () => {},
    } as unknown as AgentSideConnection;
    return new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
  }

  function createResultMessage(overrides: {
    subtype: "success" | "error_during_execution";
    stop_reason: string | null;
    is_error: boolean;
    result?: string;
    errors?: string[];
  }) {
    return {
      type: "result" as const,
      subtype: overrides.subtype,
      stop_reason: overrides.stop_reason,
      is_error: overrides.is_error,
      result: overrides.result ?? "",
      errors: overrides.errors ?? [],
      duration_ms: 0,
      duration_api_ms: 0,
      num_turns: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: randomUUID(),
      session_id: "test-session",
    };
  }

  // Two-turn generator: turn 1 yields the caller-supplied `firstTurn`
  // messages (including a trailing idle that the drain must consume).
  // Turn 2 yields a clean success + idle, used to verify the next prompt
  // sees real messages rather than the stale idle.
  function injectTwoTurnSession(agent: ClaudeAcpAgent, firstTurn: unknown[]) {
    const input = new Pushable<any>();
    const interrupt = vi.fn(async () => {});
    const close = vi.fn();
    async function* messageGenerator() {
      const iter = input[Symbol.asyncIterator]();

      const first = await iter.next();
      if (!first.done && first.value) {
        yield {
          type: "user",
          message: first.value.message,
          parent_tool_use_id: null,
          uuid: first.value.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      yield* firstTurn;

      const second = await iter.next();
      if (!second.done && second.value) {
        yield {
          type: "user",
          message: second.value.message,
          parent_tool_use_id: null,
          uuid: second.value.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      yield createResultMessage({ subtype: "success", stop_reason: null, is_error: false });
      yield { type: "system", subtype: "session_state_changed", state: "idle" };
    }
    const gen = Object.assign(messageGenerator(), { interrupt, close });
    agent.sessions["test-session"] = {
      query: gen as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");
    return { interrupt };
  }

  it("drains a failed turn's trailing idle so the next prompt is not short-circuited", async () => {
    const agent = createMockAgent();
    const { interrupt } = injectTwoTurnSession(agent, [
      createResultMessage({
        subtype: "success",
        stop_reason: "end_turn",
        is_error: true,
        result: "boom",
      }),
      // Trailing idle from the failed turn. Without draining, the next
      // prompt's first query.next() would consume this and short-circuit
      // to end_turn with zero usage (issue #654).
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    await expect(
      agent.prompt({
        sessionId: "test-session",
        prompt: [{ type: "text", text: "first" }],
      }),
    ).rejects.toThrow();

    expect(interrupt).toHaveBeenCalled();

    const second = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "second" }],
    });
    expect(second.stopReason).toBe("end_turn");
    expect(second.usage?.inputTokens).toBe(10);
    expect(second.usage?.outputTokens).toBe(5);
  });

  it("cancels all queued pending prompts when a turn errors", async () => {
    const agent = createMockAgent();
    injectTwoTurnSession(agent, [
      createResultMessage({
        subtype: "success",
        stop_reason: "end_turn",
        is_error: true,
        result: "boom",
      }),
      { type: "system", subtype: "session_state_changed", state: "idle" },
    ]);

    // Simulate two prompts already queued behind the running turn. Both
    // resolvers should fire with `true` (cancelled) when the running
    // prompt errors, and the map should be cleared.
    const session = agent.sessions["test-session"];
    let resolveA!: (cancelled: boolean) => void;
    let resolveB!: (cancelled: boolean) => void;
    const pendingA = new Promise<boolean>((r) => (resolveA = r));
    const pendingB = new Promise<boolean>((r) => (resolveB = r));
    session.pendingMessages.set("uuid-a", { resolve: resolveA, order: 0 });
    session.pendingMessages.set("uuid-b", { resolve: resolveB, order: 1 });

    await expect(
      agent.prompt({
        sessionId: "test-session",
        prompt: [{ type: "text", text: "first" }],
      }),
    ).rejects.toThrow();

    await expect(pendingA).resolves.toBe(true);
    await expect(pendingB).resolves.toBe(true);
    expect(session.pendingMessages.size).toBe(0);
  });
});

describe("session/cancel wedge recovery (issue #680)", () => {
  function createMockAgent() {
    const mockClient = {
      sessionUpdate: async () => {},
    } as unknown as AgentSideConnection;
    return new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
  }

  // Generator that replays the prompt's user message and then blocks forever,
  // simulating the SDK wedged in a `TaskOutput { block: true }` poll against a
  // hung background task. `interrupt()` is a no-op — it does NOT unblock the
  // generator, matching the SDK behavior described in the issue.
  function injectWedgedSession(agent: ClaudeAcpAgent, opts: { interruptUnblocks?: boolean } = {}) {
    const input = new Pushable<any>();
    const interrupt = vi.fn(async () => {});
    const close = vi.fn();
    // A promise the wedged poll awaits. When `interruptUnblocks` is set, the
    // mocked interrupt() resolves it so the generator yields a trailing idle —
    // the normal, healthy interrupt path.
    let releaseBlock!: () => void;
    const blocked = new Promise<void>((resolve) => {
      releaseBlock = resolve;
    });
    if (opts.interruptUnblocks) {
      interrupt.mockImplementation(async () => {
        releaseBlock();
      });
    }

    async function* messageGenerator() {
      const iter = input[Symbol.asyncIterator]();
      const first = await iter.next();
      if (!first.done && first.value) {
        yield {
          type: "user",
          message: first.value.message,
          parent_tool_use_id: null,
          uuid: first.value.uuid,
          session_id: "test-session",
          isReplay: true,
        };
      }
      // Wedge: never yield again unless interrupt() releases us.
      await blocked;
      yield { type: "system", subtype: "session_state_changed", state: "idle" };
    }

    const gen = Object.assign(messageGenerator(), { interrupt, close });
    agent.sessions["test-session"] = {
      query: gen as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn: new OffTurnFollowupCollector("test-session", async () => {}, {
        log: () => {},
        error: () => {},
      }),
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    startReaderForTest(agent, "test-session");
    return { interrupt };
  }

  it("resolves the pending prompt with cancelled when the SDK never yields after interrupt", async () => {
    const agent = createMockAgent();
    // Shrink the grace period so the test doesn't wait the production default.
    agent.forceCancelGraceMs = 20;
    const { interrupt } = injectWedgedSession(agent);

    const promptPromise = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "run cargo test" }],
    });

    // Let the loop consume the replay and block on the wedged query.next().
    await new Promise((r) => setTimeout(r, 5));

    await agent.cancel({ sessionId: "test-session" });

    const response = await promptPromise;
    expect(response.stopReason).toBe("cancelled");
    expect(interrupt).toHaveBeenCalled();
  });

  it("returns cancelled through the normal idle path without waiting the grace period when interrupt works", async () => {
    const agent = createMockAgent();
    // Large grace so that if the test ever falls through to the backstop it
    // would hang past the test timeout instead of passing by accident.
    agent.forceCancelGraceMs = 60_000;
    const { interrupt } = injectWedgedSession(agent, { interruptUnblocks: true });

    const promptPromise = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "run cargo test" }],
    });

    await new Promise((r) => setTimeout(r, 5));

    await agent.cancel({ sessionId: "test-session" });

    const response = await promptPromise;
    expect(response.stopReason).toBe("cancelled");
    expect(interrupt).toHaveBeenCalled();
    // Backstop timer must have been cleared so it can't fire later.
    expect(agent.sessions["test-session"].forceCancelTimer).toBeUndefined();
  });

  it("does not arm the backstop when no prompt is running", async () => {
    const agent = createMockAgent();
    injectWedgedSession(agent);

    await agent.cancel({ sessionId: "test-session" });

    const session = agent.sessions["test-session"];
    expect(session.cancelled).toBe(true);
    expect(session.forceCancelTimer).toBeUndefined();
  });

  it("does not reset the force-cancel floor on repeated cancels", async () => {
    const agent = createMockAgent();
    // Long floor so the timer handle stays observable across both cancels.
    agent.forceCancelGraceMs = 60_000;
    injectWedgedSession(agent);

    const promptPromise = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "run cargo test" }],
    });
    await new Promise((r) => setTimeout(r, 5));

    await agent.cancel({ sessionId: "test-session" });
    const firstTimer = agent.sessions["test-session"].forceCancelTimer;
    expect(firstTimer).toBeDefined();

    await agent.cancel({ sessionId: "test-session" });
    // Same handle: the second cancel did not clear-and-rearm (which would push
    // the floor out). The deadline stays anchored to the first cancel.
    expect(agent.sessions["test-session"].forceCancelTimer).toBe(firstTimer);

    // Clean up the wedged prompt + long timer.
    await agent.closeSession({ sessionId: "test-session" });
    await expect(promptPromise).resolves.toEqual({ stopReason: "cancelled" });
  });

  it("resolves an in-flight wedged prompt immediately when the session is closed", async () => {
    const agent = createMockAgent();
    // Large floor: if closeSession relied on the force-cancel timer this would
    // hang past the test timeout. Teardown must wake the loop via
    // cancelController instead.
    agent.forceCancelGraceMs = 60_000;
    injectWedgedSession(agent);

    const promptPromise = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "run cargo test" }],
    });
    await new Promise((r) => setTimeout(r, 5));

    await agent.closeSession({ sessionId: "test-session" });

    await expect(promptPromise).resolves.toEqual({ stopReason: "cancelled" });
    expect(agent.sessions["test-session"]).toBeUndefined();
  });
});

describe("streamEventToAcpNotifications", () => {
  it("treats `ping` keep-alive events as no-ops without logging to stderr", () => {
    const errors: unknown[][] = [];
    const logger = {
      log: () => {},
      error: (...args: unknown[]) => {
        errors.push(args);
      },
    };
    const pingMessage = {
      type: "stream_event",
      parent_tool_use_id: null,
      uuid: randomUUID(),
      session_id: "test-session",
      // The SDK's typed `BetaRawMessageStreamEvent` union doesn't include
      // `ping`, but the API emits it on the wire and the SDK passes it
      // through. Cast through `unknown` to feed the realistic runtime shape.
      event: { type: "ping" } as unknown,
    } as Parameters<typeof streamEventToAcpNotifications>[0];

    const result = streamEventToAcpNotifications(
      pingMessage,
      "test-session",
      {},
      { sessionUpdate: async () => {} } as unknown as Parameters<
        typeof streamEventToAcpNotifications
      >[3],
      logger,
    );

    expect(result).toEqual([]);
    expect(errors).toEqual([]);
  });

  it("attaches the supplied messageId to streamed text chunks", () => {
    const messageId = randomUUID();
    const message = {
      type: "stream_event",
      parent_tool_use_id: null,
      uuid: randomUUID(),
      session_id: "test-session",
      event: {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "hello" },
      },
    } as Parameters<typeof streamEventToAcpNotifications>[0];

    const result = streamEventToAcpNotifications(
      message,
      "test",
      {},
      {} as AgentSideConnection,
      console,
      { messageId },
    );

    expect(result).toEqual([
      {
        sessionId: "test",
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "hello" },
          messageId,
        },
      },
    ]);
  });
});

describe("toAcpNotifications messageId", () => {
  const messageId = "11111111-2222-3333-4444-555555555555";

  it("sets messageId on agent message chunks from string content", () => {
    const result = toAcpNotifications(
      "hello world",
      "assistant",
      "test",
      {},
      {} as AgentSideConnection,
      console,
      { messageId },
    );

    expect(result).toEqual([
      {
        sessionId: "test",
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "hello world" },
          messageId,
        },
      },
    ]);
  });

  it("sets messageId on user message chunks and thought chunks", () => {
    const userResult = toAcpNotifications(
      [{ type: "text", text: "hi" }],
      "user",
      "test",
      {},
      {} as AgentSideConnection,
      console,
      { messageId },
    );
    expect(userResult[0].update).toMatchObject({
      sessionUpdate: "user_message_chunk",
      messageId,
    });

    const thoughtResult = toAcpNotifications(
      [{ type: "thinking", thinking: "hmm", signature: "" }],
      "assistant",
      "test",
      {},
      {} as AgentSideConnection,
      console,
      { messageId },
    );
    expect(thoughtResult[0].update).toMatchObject({
      sessionUpdate: "agent_thought_chunk",
      messageId,
    });
  });

  it("omits messageId when none is supplied", () => {
    const result = toAcpNotifications(
      "hello",
      "assistant",
      "test",
      {},
      {} as AgentSideConnection,
      console,
    );
    expect(result[0].update).not.toHaveProperty("messageId");
  });

  it("never sets messageId on non-chunk updates (tool_call)", () => {
    const result = toAcpNotifications(
      [
        {
          type: "tool_use",
          id: "toolu_abc",
          name: "Read",
          input: { file_path: "/tmp/x" },
        },
      ],
      "assistant",
      "test",
      {},
      {} as AgentSideConnection,
      console,
      { messageId, registerHooks: false },
    );
    expect(result[0].update.sessionUpdate).toBe("tool_call");
    expect(result[0].update).not.toHaveProperty("messageId");
  });
});

describe("messageIdForGrouping", () => {
  it("uses the Anthropic API message id for assistant messages", () => {
    const message = {
      type: "assistant",
      uuid: "de242400-cdb3-4af7-9856-d3b114b20af9",
      message: { id: "msg_018DQGVuZbGYwVnvDakAP9Do", role: "assistant" },
    };
    // The API id is identical at message_start, on the consolidated message,
    // and in the persisted transcript — so it stays stable across replay,
    // unlike the per-message uuid.
    expect(messageIdForGrouping(message)).toBe("msg_018DQGVuZbGYwVnvDakAP9Do");
  });

  it("falls back to the uuid for assistant messages without an API id", () => {
    const message = {
      type: "assistant",
      uuid: "de242400-cdb3-4af7-9856-d3b114b20af9",
      message: { role: "assistant" },
    };
    expect(messageIdForGrouping(message)).toBe("de242400-cdb3-4af7-9856-d3b114b20af9");
  });

  it("uses the uuid for user messages (they carry no API id and aren't streamed)", () => {
    const message = {
      type: "user",
      uuid: "11111111-2222-3333-4444-555555555555",
      message: { id: "msg_should_be_ignored", role: "user" },
    };
    expect(messageIdForGrouping(message)).toBe("11111111-2222-3333-4444-555555555555");
  });

  it("returns undefined when there is no usable id", () => {
    expect(messageIdForGrouping({ type: "system", message: {} })).toBeUndefined();
    expect(messageIdForGrouping({ type: "assistant", uuid: "", message: {} })).toBeUndefined();
  });
});

describe("session reader (issue #336)", () => {
  // Single-consumer producer of SDK messages for the per-session reader.
  // Tracks concurrent calls to assert the single-consumer invariant.
  class QueryStub {
    private queue: any[] = [];
    private waiter: ((r: IteratorResult<any, void>) => void) | null = null;
    private rejecter: ((err: unknown) => void) | null = null;
    private closed = false;
    private throwOnNextErr: unknown = undefined;
    /** Highest number of `next()` calls that were simultaneously in flight. */
    public maxConcurrentReads = 0;
    private inFlight = 0;
    /** Counts every `next()` invocation, including those that resolved
     *  synchronously from the buffer. */
    public nextCallCount = 0;

    /** True when the producer queue is empty and the reader is parked on
     *  next(). Tests use this to know the reader has caught up to the
     *  most recent push before asserting reader-driven state. */
    isIdle(): boolean {
      return this.queue.length === 0 && this.waiter !== null;
    }

    push(item: any): void {
      if (this.waiter) {
        const w = this.waiter;
        this.waiter = null;
        this.rejecter = null;
        w({ value: item, done: false });
      } else {
        this.queue.push(item);
      }
    }
    close(): void {
      this.closed = true;
      if (this.waiter) {
        const w = this.waiter;
        this.waiter = null;
        this.rejecter = null;
        w({ value: undefined as any, done: true });
      }
    }
    throwOnNext(err: unknown): void {
      this.throwOnNextErr = err;
      if (this.rejecter) {
        const r = this.rejecter;
        this.waiter = null;
        this.rejecter = null;
        r(err);
      }
    }
    next(): Promise<IteratorResult<any, void>> {
      this.nextCallCount++;
      this.inFlight++;
      if (this.inFlight > this.maxConcurrentReads) {
        this.maxConcurrentReads = this.inFlight;
      }
      const finish = <T>(v: T): T => {
        this.inFlight--;
        return v;
      };
      if (this.throwOnNextErr !== undefined) {
        const err = this.throwOnNextErr;
        this.throwOnNextErr = undefined;
        this.inFlight--;
        return Promise.reject(err);
      }
      if (this.queue.length > 0) {
        const value = this.queue.shift()!;
        this.inFlight--;
        return Promise.resolve({ value, done: false });
      }
      if (this.closed) {
        this.inFlight--;
        return Promise.resolve({ value: undefined as any, done: true });
      }
      return new Promise<IteratorResult<any, void>>((resolve, reject) => {
        this.waiter = (r) => {
          finish(undefined);
          resolve(r);
        };
        this.rejecter = (e) => {
          finish(undefined);
          reject(e);
        };
      });
    }
    // The session reader doesn't call these but production paths might.
    interrupt(): Promise<void> {
      return Promise.resolve();
    }
    closeQuery(): void {
      this.closed = true;
    }
  }

  function createCaptureClient() {
    const sessionUpdates: Array<{ sessionId: string; update: any }> = [];
    const extNotifications: Array<{ method: string; params: any }> = [];
    const client = {
      sessionUpdate: vi.fn(async (n: any) => {
        sessionUpdates.push(n);
      }),
      extNotification: vi.fn(async (method: string, params: any) => {
        extNotifications.push({ method, params });
      }),
    } as unknown as AgentSideConnection;
    return { client, sessionUpdates, extNotifications };
  }

  function buildSession(
    sessionId: string,
    agent: ClaudeAcpAgent,
    query: QueryStub,
    input: Pushable<any>,
    opts: { emitRawSDKMessages?: boolean; useRealFollowup?: boolean } = {},
  ) {
    // Default emitter is a no-op; tests that exercise followup forwarding
    // replace `session.offTurn` with a collector bound to a capturing
    // emitter (see the followup-forwarding tests below). With
    // useRealFollowup the collector is bound to the production #emitFollowup
    // so the end-to-end render path is exercised.
    const emitter = opts.useRealFollowup
      ? (msgs: any[], result: any) =>
          (
            agent as unknown as {
              emitFollowupForTest(id: string, m: any[], r: any): Promise<void>;
            }
          ).emitFollowupForTest(sessionId, msgs, result)
      : async () => {};
    const offTurn = new OffTurnFollowupCollector(sessionId, emitter, {
      log: () => {},
      error: () => {},
    });
    const session = {
      query: query as any,
      input,
      cancelled: false,
      cwd: "/test",
      sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
      settingsManager: { dispose: vi.fn() } as any,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelInfos: [],
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: opts.emitRawSDKMessages ?? false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      turnQueue: new TurnQueue(),
      offTurn,
      readerDone: Promise.resolve(),
      readerSideEffects: Promise.resolve(),
    };
    agent.sessions[sessionId] = session as any;
    startReaderForTest(agent, sessionId);
    return session;
  }

  function createAgent(client: AgentSideConnection) {
    return new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });
  }

  it("forwards task_started using description (not summary) for the label", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    query.push({
      type: "system",
      subtype: "task_started",
      task_id: "t1",
      description: "ship the docs",
      uuid: "u",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(sessionUpdates.length).toBe(1));
    expect(sessionUpdates[0].update.sessionUpdate).toBe("agent_message_chunk");
    // Wrapped in blank lines so back-to-back lifecycle chunks and following
    // agent text don't fuse into one markdown paragraph.
    expect(sessionUpdates[0].update.content.text).toBe("\n\n[task t1] started: ship the docs\n\n");

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("forwards task_notification with the SDK status (no fallback to 'completed')", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    query.push({
      type: "system",
      subtype: "task_notification",
      task_id: "t9",
      status: "failed",
      summary: "build broke",
      output_file: "/tmp/9",
      uuid: "u",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(sessionUpdates.length).toBe(1));
    expect(sessionUpdates[0].update.content.text).toContain("[task t9] failed: build broke");
    expect(sessionUpdates[0].update.content.text).toContain("output: /tmp/9");

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("wraps lifecycle chunks in blank lines so back-to-back tasks stay separated", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    // Two notifications landing back-to-back (e.g. two background tasks
    // finishing together) are the case that regressed: clients concatenate
    // chunk text raw, so without blank-line wrapping they fuse into one line.
    query.push({
      type: "system",
      subtype: "task_notification",
      task_id: "t1",
      status: "completed",
      summary: "first",
      uuid: "u1",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "task_notification",
      task_id: "t2",
      status: "completed",
      summary: "second",
      uuid: "u2",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(sessionUpdates.length).toBe(2));
    // Each chunk is wrapped, so concatenating them yields a blank-line
    // (paragraph) separator between the two tasks rather than a fused run.
    const concatenated = sessionUpdates.map((u) => u.update.content.text).join("");
    expect(concatenated).toContain("[task t1] completed: first\n\n");
    expect(concatenated).toContain("\n\n[task t2] completed: second");
    expect(concatenated).not.toContain("first[task t2]");

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("suppresses lifecycle forward when skip_transcript=true", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    query.push({
      type: "system",
      subtype: "task_notification",
      task_id: "t2",
      status: "completed",
      summary: "",
      output_file: "",
      skip_transcript: true,
      uuid: "u",
      session_id: "s1",
    });
    // Use a no-op message after the suppressed one to give the reader
    // something to ack before we assert.
    query.push({
      type: "system",
      subtype: "task_started",
      task_id: "t3",
      description: "follow-up",
      uuid: "u2",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(sessionUpdates.length).toBe(1));
    expect(sessionUpdates[0].update.content.text).toContain("[task t3] started");

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("emits raw SDK messages uniformly for both in-turn and off-turn", async () => {
    const { client, extNotifications } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input, { emitRawSDKMessages: true });

    // Off-turn lifecycle: reader should raw-emit.
    query.push({
      type: "system",
      subtype: "task_started",
      task_id: "t1",
      description: "off-turn",
      uuid: "u",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(extNotifications.length).toBe(1));
    expect(extNotifications[0].method).toBe("_claude/sdkMessage");
    expect(extNotifications[0].params.sessionId).toBe("s1");

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("does not double-emit raw SDK messages for in-turn traffic", async () => {
    // The reader is the only raw-emit site; prompt() no longer raw-emits.
    // Verify that each in-turn message produces exactly one
    // `_claude/sdkMessage` notification.
    const { client, extNotifications } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input, { emitRawSDKMessages: true });

    const promptDone = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "hi" }],
    });

    // Replay + result + idle: three in-turn messages.
    query.push({
      type: "user",
      message: { role: "user", content: "hi" },
      parent_tool_use_id: null,
      isReplay: true,
      uuid: "ur",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: "ures",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
    });

    await promptDone;
    await session.readerSideEffects;
    // Three messages → exactly three raw-emit notifications, not six.
    expect(extNotifications.length).toBe(3);
    for (const n of extNotifications) {
      expect(n.method).toBe("_claude/sdkMessage");
    }

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("emits raw SDK messages in FIFO order (single side-effect chain)", async () => {
    const { client, extNotifications } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input, { emitRawSDKMessages: true });

    // Push several off-turn lifecycle messages; each is raw-emitted on the
    // readerSideEffects chain. Order among raw emits must match push order.
    for (let i = 0; i < 5; i++) {
      query.push({
        type: "system",
        subtype: "task_started",
        task_id: `t${i}`,
        description: `task ${i}`,
        uuid: `u${i}`,
        session_id: "s1",
      });
    }

    await vi.waitFor(() => expect(extNotifications.length).toBe(5));
    const ids = extNotifications.map((n) => n.params.message.task_id);
    expect(ids).toEqual(["t0", "t1", "t2", "t3", "t4"]);

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("renders a real followup end-to-end via #emitFollowup (content + usage_update)", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    // useRealFollowup binds the off-turn collector to the production
    // #emitFollowup so the actual render path runs (not a capturing stub).
    const session = buildSession("s1", agent, query, input, { useRealFollowup: true });

    // A followup: an assistant message carrying a tool_use block (text is
    // filtered out since it would be streamed), closed by a task-notification
    // result.
    query.push({
      type: "assistant",
      message: {
        role: "assistant",
        model: "claude-sonnet-4-6",
        content: [
          { type: "text", text: "streamed-elsewhere" },
          { type: "tool_use", id: "tu1", name: "Bash", input: { command: "ls" } },
        ],
        usage: {
          input_tokens: 12,
          output_tokens: 8,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
      parent_tool_use_id: null,
      uuid: "a1",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "done",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0.25,
      usage: {
        input_tokens: 12,
        output_tokens: 8,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      origin: { kind: "task-notification" },
      uuid: "r1",
      session_id: "s1",
    });

    await vi.waitFor(() => {
      expect(query.isIdle()).toBe(true);
      expect(session.offTurn.inspect().state).toBe("idle");
    });
    // Drain the detached side-effect chain that #emitFollowup runs on.
    await session.readerSideEffects;

    // The tool_use rendered as a tool_call, and a usage_update closed the
    // followup with the task-notification origin in _meta.
    const toolCalls = sessionUpdates.filter((u) => u.update.sessionUpdate === "tool_call");
    expect(toolCalls.length).toBeGreaterThanOrEqual(1);
    const usage = sessionUpdates.filter((u) => u.update.sessionUpdate === "usage_update");
    expect(usage.length).toBe(1);
    expect(usage[0].update.used).toBe(20); // 12 + 8 (+0 +0)
    expect(usage[0].update.cost.amount).toBe(0.25);
    expect(usage[0].update._meta?.["_claude/origin"]).toEqual({ kind: "task-notification" });
    // The followup must not touch the user-turn accumulator.
    expect(session.accumulatedUsage.inputTokens).toBe(0);
    expect(session.accumulatedUsage.outputTokens).toBe(0);

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("forwards a followup's assembled text when no content_block_delta streamed (non-streaming gateway)", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input, { useRealFollowup: true });

    // A non-streaming gateway delivers the followup as one assembled assistant
    // message with no preceding deltas. The text block is the only copy, so it
    // must be forwarded rather than filtered. Mirrors #757 for the off-turn path.
    query.push({
      type: "assistant",
      message: {
        // A non-streaming gateway still stamps the assembled block with an API
        // id. Include it so the dedup lookup runs against a real id (absent
        // from streamedTextIds) rather than passing only because the set is
        // empty — this guards the id-keying path, not just the empty-set case.
        id: "msg-nonstream-1",
        role: "assistant",
        model: "claude-sonnet-4-6",
        content: [{ type: "text", text: "the followup answer" }],
        usage: {
          input_tokens: 5,
          output_tokens: 3,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
      parent_tool_use_id: null,
      uuid: "a1",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "done",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0.1,
      usage: {
        input_tokens: 5,
        output_tokens: 3,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      origin: { kind: "task-notification" },
      uuid: "r1",
      session_id: "s1",
    });

    await vi.waitFor(() => {
      expect(query.isIdle()).toBe(true);
      expect(session.offTurn.inspect().state).toBe("idle");
    });
    await session.readerSideEffects;

    const texts = sessionUpdates
      .filter((u) => u.update.sessionUpdate === "agent_message_chunk")
      .map((u) => u.update.content.text);
    expect(texts).toContain("the followup answer");

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("does not re-emit a followup's text that already streamed via content_block_delta", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input, { useRealFollowup: true });

    // Normal streaming: the followup's deltas arrive, then the consolidated
    // assistant message repeats the same text. The assembled block must be
    // dropped so the client doesn't see it twice.
    query.push({
      type: "stream_event",
      event: { type: "message_start", message: { id: "msg-fu", usage: {} } },
      parent_tool_use_id: null,
      uuid: "s0",
      session_id: "s1",
    });
    query.push({
      type: "stream_event",
      event: {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "streamed answer" },
      },
      parent_tool_use_id: null,
      uuid: "s1d",
      session_id: "s1",
    });
    query.push({
      type: "assistant",
      message: {
        id: "msg-fu",
        role: "assistant",
        model: "claude-sonnet-4-6",
        content: [{ type: "text", text: "streamed answer" }],
        usage: {
          input_tokens: 5,
          output_tokens: 3,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
      parent_tool_use_id: null,
      uuid: "a1",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "done",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0.1,
      usage: {
        input_tokens: 5,
        output_tokens: 3,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      origin: { kind: "task-notification" },
      uuid: "r1",
      session_id: "s1",
    });

    await vi.waitFor(() => {
      expect(query.isIdle()).toBe(true);
      expect(session.offTurn.inspect().state).toBe("idle");
    });
    await session.readerSideEffects;

    // "streamed answer" reaches the client once (via the delta), not twice.
    const occurrences = sessionUpdates
      .filter((u) => u.update.sessionUpdate === "agent_message_chunk")
      .map((u) => u.update.content.text)
      .filter((t) => t === "streamed answer");
    expect(occurrences.length).toBe(1);

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("forwards an autonomous task-notification followup out-of-turn", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    // Wire the off-turn collector to call the production #emitFollowup via
    // an internal test-only adapter. Easier: replace the offTurn with one
    // bound to a directly-spy emitter that captures.
    const flushed: Array<{ msgs: any[]; result: any }> = [];
    const session = buildSession("s1", agent, query, input);
    session.offTurn = new OffTurnFollowupCollector(
      "s1",
      async (msgs, result) => {
        flushed.push({ msgs, result });
      },
      { log: () => {}, error: () => {} },
    );

    query.push({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: "followup body" }] },
      parent_tool_use_id: null,
      uuid: "u1",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "done",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0.1,
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      origin: { kind: "task-notification" },
      uuid: "u2",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(flushed.length).toBe(1));
    expect(flushed[0].msgs.length).toBe(1);
    expect(flushed[0].msgs[0].type).toBe("assistant");
    expect(flushed[0].result.origin.kind).toBe("task-notification");
    expect(sessionUpdates.length).toBe(0); // followup is forwarded via the bound emitter, not sessionUpdate

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("does not contaminate the next user prompt's usage_update with a prior followup", async () => {
    // The off-turn collector drains and forwards followups via its
    // own emitter; the next prompt() starts with a fresh
    // accumulatedUsage and never sees the followup messages in its loop.
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input);

    // Drive a followup off-turn.
    query.push({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: "fu" }] },
      parent_tool_use_id: null,
      uuid: "u1",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 1,
      usage: {
        input_tokens: 1000,
        output_tokens: 500,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      origin: { kind: "task-notification" },
      uuid: "u2",
      session_id: "s1",
    });

    // Wait for the reader to drain everything we pushed and park on next().
    // Asserting on `session.offTurn.inspect().state === 'idle'` alone is
    // not enough: the collector starts in 'idle' too, so the assertion can
    // pass before the reader has consumed any message. Reader-parked + the
    // collector having returned to idle is what we actually need.
    await vi.waitFor(() => {
      expect(query.isIdle()).toBe(true);
      expect(session.offTurn.inspect().state).toBe("idle");
    });

    // Now run a real user turn. The reader should not replay any
    // followup-tokens into accumulatedUsage.
    const userTurnReplay = {
      type: "user",
      message: { role: "user", content: "hi" },
      parent_tool_use_id: null,
      isReplay: true,
      uuid: "u3",
      session_id: "s1",
    };
    const turnResult = {
      type: "result",
      subtype: "success",
      is_error: false,
      result: "answer",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0.05,
      usage: {
        input_tokens: 7,
        output_tokens: 3,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: "u4",
      session_id: "s1",
    };
    const idle = {
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
    };
    // Push these so they're ready when prompt() starts.
    query.push(userTurnReplay);
    query.push(turnResult);
    query.push(idle);

    const r = await agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "hi" }],
    });
    expect(r.stopReason).toBe("end_turn");
    expect(session.accumulatedUsage.inputTokens).toBe(7);
    expect(session.accumulatedUsage.outputTokens).toBe(3);
    // No "Please run /login" tripwire from a stale result, no usage_update
    // with the followup's 1000+500.
    const usageUpdates = sessionUpdates.filter((u) => u.update.sessionUpdate === "usage_update");
    for (const u of usageUpdates) {
      expect(u.update.used).not.toBe(1500);
    }

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("does not throw authRequired for a followup result.result containing 'Please run /login'", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const flushed: any[] = [];
    const session = buildSession("s1", agent, query, input);
    session.offTurn = new OffTurnFollowupCollector(
      "s1",
      async (msgs, result) => {
        flushed.push({ msgs, result });
      },
      { log: () => {}, error: () => {} },
    );

    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "Please run /login",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      origin: { kind: "task-notification" },
      uuid: "u",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(flushed.length).toBe(1));
    // The reader stayed up; no throw escaped.
    expect(session.offTurn.inspect().state).toBe("idle");

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("forwards lifecycle in the middle of a followup-candidate without breaking the FSM", async () => {
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const flushed: any[] = [];
    const session = buildSession("s1", agent, query, input);
    session.offTurn = new OffTurnFollowupCollector(
      "s1",
      async (msgs, result) => {
        flushed.push({ msgs, result });
      },
      { log: () => {}, error: () => {} },
    );

    // Start a followup candidate.
    query.push({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: "a" }] },
      parent_tool_use_id: null,
      uuid: "u1",
      session_id: "s1",
    });
    // Lifecycle slips in mid-candidate — should be forwarded immediately,
    // not added to the buffer.
    query.push({
      type: "system",
      subtype: "task_started",
      task_id: "t2",
      description: "concurrent",
      uuid: "u2",
      session_id: "s1",
    });
    // Close the candidate with a followup result.
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      origin: { kind: "task-notification" },
      uuid: "u3",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(flushed.length).toBe(1));
    // The lifecycle was forwarded as a sessionUpdate.
    expect(sessionUpdates.some((u) => u.update.content?.text?.includes("[task t2] started"))).toBe(
      true,
    );
    // The buffer contained only the assistant, not the lifecycle.
    expect(flushed[0].msgs.length).toBe(1);
    expect(flushed[0].msgs[0].type).toBe("assistant");

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("does not let a slow lifecycle sessionUpdate block a later prompt", async () => {
    let releaseLifecycle!: () => void;
    const lifecycleBlocked = new Promise<void>((resolve) => {
      releaseLifecycle = resolve;
    });
    const client = {
      sessionUpdate: vi.fn(async () => {
        await lifecycleBlocked;
      }),
      extNotification: vi.fn(async () => {}),
    } as unknown as AgentSideConnection;
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    query.push({
      type: "system",
      subtype: "task_notification",
      task_id: "t-slow",
      status: "completed",
      summary: "slow client",
      uuid: "u-slow",
      session_id: "s1",
    });
    await vi.waitFor(() => expect(client.sessionUpdate).toHaveBeenCalledTimes(1));

    const promptDone = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "hi" }],
    });
    query.push({
      type: "user",
      message: { role: "user", content: "hi" },
      parent_tool_use_id: null,
      isReplay: true,
      uuid: "u1",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 2,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: "u2",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
      session_id: "s1",
    });

    await expect(promptDone).resolves.toMatchObject({ stopReason: "end_turn" });
    releaseLifecycle();
    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("clears queued turn leftovers after a prompt error", async () => {
    let releaseFirstUpdate!: () => void;
    let blockFirstUpdate = true;
    const firstUpdateBlocked = new Promise<void>((resolve) => {
      releaseFirstUpdate = resolve;
    });
    const { client } = createCaptureClient();
    vi.mocked(client.sessionUpdate).mockImplementation(async () => {
      if (blockFirstUpdate) {
        blockFirstUpdate = false;
        await firstUpdateBlocked;
      }
    });
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input);

    const firstPrompt = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "first" }],
    });
    query.push({
      type: "system",
      subtype: "local_command_output",
      content: "blocking update",
      uuid: "u-block",
      session_id: "s1",
    });
    await vi.waitFor(() => expect(client.sessionUpdate).toHaveBeenCalledTimes(1));
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "Please run /login",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 100,
        output_tokens: 100,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: "u-stale-result",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
      uuid: "u-stale-idle",
      session_id: "s1",
    });
    await vi.waitFor(() => expect(session.turnQueue.size()).toBeGreaterThan(0));
    releaseFirstUpdate();
    await expect(firstPrompt).rejects.toThrow();
    expect(session.turnQueue.size()).toBe(0);

    const secondPrompt = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "second" }],
    });
    query.push({
      type: "user",
      message: { role: "user", content: "second" },
      parent_tool_use_id: null,
      isReplay: true,
      uuid: "u-next-user",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 4,
        output_tokens: 2,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: "u-next-result",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
      uuid: "u-next-idle",
      session_id: "s1",
    });

    await expect(secondPrompt).resolves.toMatchObject({ stopReason: "end_turn" });
    expect(session.accumulatedUsage.inputTokens).toBe(4);
    expect(session.accumulatedUsage.outputTokens).toBe(2);

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("discards orphan off-turn messages followed by an idle", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input);

    query.push({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: "x" }] },
      parent_tool_use_id: null,
      uuid: "u",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
      uuid: "u2",
      session_id: "s1",
    });

    // Wait until the reader has drained the queue and parked on next();
    // only then is the buffer state reliable.
    await vi.waitFor(() => {
      expect(query.isIdle()).toBe(true);
      expect(session.offTurn.inspect().state).toBe("idle");
    });
    expect(session.offTurn.inspect().bufferSize).toBe(0);

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("discards off-turn messages followed by a non-followup result", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const flushed: any[] = [];
    const session = buildSession("s1", agent, query, input);
    session.offTurn = new OffTurnFollowupCollector(
      "s1",
      async (msgs, result) => {
        flushed.push({ msgs, result });
      },
      { log: () => {}, error: () => {} },
    );

    query.push({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: "x" }] },
      parent_tool_use_id: null,
      uuid: "u",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      // No origin field → not a followup.
      uuid: "u2",
      session_id: "s1",
    });

    await vi.waitFor(() => {
      expect(query.isIdle()).toBe(true);
      expect(session.offTurn.inspect().state).toBe("idle");
    });
    expect(flushed.length).toBe(0);

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("cancellation aftermath is discarded by the reader, not replayed into the next turn", async () => {
    // After a cancel() interrupt, the SDK can emit a tail of messages
    // (a result with the interrupted turn and an idle). The reader sees
    // them off-turn because prompt() already returned. The off-turn
    // collector discards them — they must not contaminate the next user
    // prompt's accumulatedUsage or stopReason.
    const { client, sessionUpdates } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input);

    // Run a turn, then send a cancelled idle so prompt() returns cancelled.
    const firstPrompt = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "first" }],
    });

    query.push({
      type: "user",
      message: { role: "user", content: "first" },
      parent_tool_use_id: null,
      isReplay: true,
      uuid: "u1",
      session_id: "s1",
    });
    // Simulate cancel mid-turn: cancelled=true + idle to end the turn.
    session.cancelled = true;
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
    });
    const first = await firstPrompt;
    expect(first.stopReason).toBe("cancelled");

    // Aftermath that the SDK might emit AFTER the interrupted turn ended.
    query.push({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: "stale" }] },
      parent_tool_use_id: null,
      uuid: "u2",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "error_during_execution",
      is_error: false,
      result: "interrupted",
      stop_reason: "interrupted",
      errors: [],
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 9999,
        output_tokens: 9999,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: "u3",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
    });

    // Wait for the reader to drain the aftermath.
    await vi.waitFor(() => {
      expect(query.isIdle()).toBe(true);
      expect(session.offTurn.inspect().state).toBe("idle");
    });

    // Reset session.cancelled and run the next user turn cleanly.
    session.cancelled = false;
    const secondPrompt = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "second" }],
    });
    query.push({
      type: "user",
      message: { role: "user", content: "second" },
      parent_tool_use_id: null,
      isReplay: true,
      uuid: "u4",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 4,
        output_tokens: 2,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: "u5",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
    });

    const second = await secondPrompt;
    expect(second.stopReason).toBe("end_turn");
    // The aftermath's 9999/9999 must not be in the second turn's usage.
    expect(session.accumulatedUsage.inputTokens).toBe(4);
    expect(session.accumulatedUsage.outputTokens).toBe(2);
    // And no usage_update was emitted for the aftermath result.
    const usageUpdates = sessionUpdates.filter((u) => u.update.sessionUpdate === "usage_update");
    for (const u of usageUpdates) {
      expect(u.update.used).not.toBe(9999 * 2);
    }

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("process-died throw in reader propagates to the active prompt as an error", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    const promptPromise = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "hi" }],
    });
    // Give the reader a chance to grab its first next() and prompt() a
    // chance to set promptRunning + start awaiting turnQueue.
    await new Promise((r) => setTimeout(r, 5));

    query.throwOnNext(new Error("process exited with code 1"));

    await expect(promptPromise).rejects.toBeInstanceOf(Error);

    // Session entry should have been cleaned up by the process-died path.
    expect(agent.sessions["s1"]).toBeUndefined();
  });

  it("tears down the session when the reader dies on a non-process-death iterator error", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    const promptPromise = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "hi" }],
    });
    await new Promise((r) => setTimeout(r, 5));

    // An error whose message matches none of the process-death substrings.
    // The reader errors the turnQueue and exits; without teardown the queue's
    // latched error would brick every future prompt() on this session.
    query.throwOnNext(new Error("Unexpected event order, got delta before message_start"));

    await expect(promptPromise).rejects.toBeInstanceOf(Error);
    // The dead reader can never feed this session again, so it must be removed
    // rather than left in the map with a permanently errored queue. See #336.
    expect(agent.sessions["s1"]).toBeUndefined();
  });

  it("returns cancelled (not an error) when a force-cancel races an iterator throw", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input);

    const promptPromise = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "hi" }],
    });
    await new Promise((r) => setTimeout(r, 5));

    // Reproduce the race where the take() rejection wins: cancel() has set
    // session.cancelled but the force-cancel timer has NOT yet aborted the
    // wake-up channel (so the `cancelled` promise never resolves), and the
    // iterator throws. The rejection propagates through Promise.race into the
    // catch, bypassing the line-level aborted check. The catch must still
    // honor the cancel contract via session.cancelled and return "cancelled"
    // rather than surfacing the trailing error as an internal error. See
    // #680/#336.
    session.cancelled = true;
    query.throwOnNext(new Error("stream has ended, this shouldn't happen"));

    await expect(promptPromise).resolves.toEqual({ stopReason: "cancelled" });
  });

  it("teardownSession awaits the reader before deleting the session entry", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);
    const readerDone = agent.sessions["s1"]!.readerDone;

    let resolved = false;
    void readerDone.then(() => {
      resolved = true;
    });

    // teardownSession is private; reach it via the public closeSession.
    const teardownPromise = (
      agent as unknown as { teardownSession(id: string): Promise<void> }
    ).teardownSession("s1");

    // The reader exits when query.close() fires from teardown; resolve it.
    query.close();
    await teardownPromise;
    expect(resolved).toBe(true);
    expect(agent.sessions["s1"]).toBeUndefined();
  });

  it("drops queued reader side effects after the session entry is replaced", async () => {
    let releaseFirstUpdate!: () => void;
    const firstUpdateBlocked = new Promise<void>((resolve) => {
      releaseFirstUpdate = resolve;
    });
    const { client, sessionUpdates } = createCaptureClient();
    vi.mocked(client.sessionUpdate).mockImplementation(async (n: any) => {
      sessionUpdates.push(n);
      if (sessionUpdates.length === 1) {
        await firstUpdateBlocked;
      }
    });
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const oldSession = buildSession("s1", agent, query, input);

    query.push({
      type: "system",
      subtype: "task_started",
      task_id: "old-1",
      description: "already running",
      uuid: "u1",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "task_started",
      task_id: "old-2",
      description: "queued stale side effect",
      uuid: "u2",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(sessionUpdates.length).toBe(1));

    oldSession.abortController.abort();
    query.close();
    delete agent.sessions["s1"];

    const replacementQuery = new QueryStub();
    const replacementInput = new Pushable<any>();
    const replacementSession = buildSession("s1", agent, replacementQuery, replacementInput);

    releaseFirstUpdate();
    await oldSession.readerSideEffects;
    await Promise.resolve();

    expect(sessionUpdates).toHaveLength(1);
    expect(sessionUpdates[0].update.content.text).toContain("[task old-1]");

    replacementSession.abortController.abort();
    replacementQuery.close();
  });

  it("teardownSession drains a pending reader side effect before deleting the session", async () => {
    let releaseUpdate!: () => void;
    let updateStarted = false;
    const { client, sessionUpdates } = createCaptureClient();
    vi.mocked(client.sessionUpdate).mockImplementation(async (n: any) => {
      sessionUpdates.push(n);
      updateStarted = true;
      await new Promise<void>((resolve) => {
        releaseUpdate = resolve;
      });
    });
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    // A lifecycle message schedules a (slow) sessionUpdate on the detached
    // readerSideEffects chain.
    query.push({
      type: "system",
      subtype: "task_started",
      task_id: "t1",
      description: "slow",
      uuid: "u1",
      session_id: "s1",
    });
    await vi.waitFor(() => expect(updateStarted).toBe(true));

    let teardownResolved = false;
    const teardownPromise = (agent as unknown as { teardownSession(id: string): Promise<void> })
      .teardownSession("s1")
      .then(() => {
        teardownResolved = true;
      });
    query.close();

    // readerDone resolves quickly (reader loop exits on close), but the
    // side-effect chain is still blocked on the slow sessionUpdate, so
    // teardown must not have resolved yet.
    await new Promise((r) => setTimeout(r, 20));
    expect(teardownResolved).toBe(false);

    releaseUpdate();
    await teardownPromise;
    expect(teardownResolved).toBe(true);
    expect(agent.sessions["s1"]).toBeUndefined();
  });

  it("stops a multi-update followup side effect if the session entry is replaced mid-emit", async () => {
    let releaseFirstUpdate!: () => void;
    const firstUpdateBlocked = new Promise<void>((resolve) => {
      releaseFirstUpdate = resolve;
    });
    const { client, sessionUpdates } = createCaptureClient();
    vi.mocked(client.sessionUpdate).mockImplementation(async (n: any) => {
      sessionUpdates.push(n);
      if (sessionUpdates.length === 1) {
        await firstUpdateBlocked;
      }
    });
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const oldSession = buildSession("s1", agent, query, input, { useRealFollowup: true });

    query.push({
      type: "assistant",
      message: {
        role: "assistant",
        model: "claude-sonnet-4-6",
        content: [{ type: "tool_use", id: "tu-stale", name: "Bash", input: { command: "pwd" } }],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
      parent_tool_use_id: null,
      uuid: "a-stale",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "done",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      origin: { kind: "task-notification" },
      uuid: "r-stale",
      session_id: "s1",
    });

    await vi.waitFor(() => expect(sessionUpdates.length).toBe(1));
    oldSession.abortController.abort();
    query.close();
    delete agent.sessions["s1"];

    const replacementQuery = new QueryStub();
    const replacementInput = new Pushable<any>();
    const replacementSession = buildSession("s1", agent, replacementQuery, replacementInput);

    releaseFirstUpdate();
    await oldSession.readerSideEffects;

    expect(sessionUpdates).toHaveLength(1);
    expect(sessionUpdates[0].update.sessionUpdate).toBe("tool_call");

    replacementSession.abortController.abort();
    replacementQuery.close();
  });

  it("drops leftover in-turn messages from the turnQueue when a turn is cancelled", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input);

    const promptPromise = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "hi" }],
    });
    await vi.waitFor(() => expect(session.promptRunning).toBe(true));

    // The reader delivers the replay (consumed by prompt), then the SDK
    // emits the cancelled idle followed by trailing aftermath. The prompt
    // returns on the idle; the aftermath must not survive into the next turn.
    query.push({
      type: "user",
      message: { role: "user", content: "hi" },
      parent_tool_use_id: null,
      isReplay: true,
      uuid: "u1",
      session_id: "s1",
    });
    session.cancelled = true;
    query.push({ type: "system", subtype: "session_state_changed", state: "idle" });
    query.push({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: "stale" }] },
      parent_tool_use_id: null,
      uuid: "u2",
      session_id: "s1",
    });

    const r = await promptPromise;
    expect(r.stopReason).toBe("cancelled");

    // The trailing aftermath arrives off-turn; its closing idle lets the
    // collector discard it.
    query.push({ type: "system", subtype: "session_state_changed", state: "idle" });
    await vi.waitFor(() => {
      expect(query.isIdle()).toBe(true);
      expect(session.offTurn.inspect().state).toBe("idle");
    });
    // No leftover in the turnQueue from the cancelled turn, and the off-turn
    // collector discarded the aftermath rather than holding it for replay.
    expect(session.turnQueue.size()).toBe(0);
    expect(session.offTurn.inspect().bufferSize).toBe(0);

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("single-consumer invariant: only one query.next() can be in flight", async () => {
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    buildSession("s1", agent, query, input);

    // Push a few messages and let the reader drain them.
    for (let i = 0; i < 5; i++) {
      query.push({
        type: "system",
        subtype: "task_started",
        task_id: `t${i}`,
        description: `task ${i}`,
        uuid: `u${i}`,
        session_id: "s1",
      });
    }
    await vi.waitFor(() => expect(query.nextCallCount).toBeGreaterThanOrEqual(5));
    expect(query.maxConcurrentReads).toBe(1);

    // Even with a concurrent prompt() and the reader cycling, only one
    // next() at a time.
    const promptDone = agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "hi" }],
    });
    query.push({
      type: "user",
      message: { role: "user", content: "hi" },
      parent_tool_use_id: null,
      isReplay: true,
      uuid: "ur",
      session_id: "s1",
    });
    query.push({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid: "urr",
      session_id: "s1",
    });
    query.push({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
    });
    await promptDone;
    expect(query.maxConcurrentReads).toBe(1);

    agent.sessions["s1"]!.abortController.abort();
    query.close();
  });

  it("resumes a queued prompt with promptRunning re-asserted when the prior turn ends without replaying it (#336)", async () => {
    // Regression: a prompt queued behind a running turn can be resumed via the
    // finally-fallback in prompt() (the prior turn ended at idle without ever
    // replaying the queued user message). That fallback sets promptRunning=false
    // and *then* resolves the waiter, so the resumed prompt must re-assert
    // promptRunning before reading the turnQueue. If it doesn't, the reader
    // routes the resumed turn's messages off-turn (where the followup collector
    // discards them) and the prompt hangs forever.
    const { client } = createCaptureClient();
    const agent = createAgent(client);
    const query = new QueryStub();
    const input = new Pushable<any>();
    const session = buildSession("s1", agent, query, input);

    const result = (usage: { input: number; output: number }, uuid: string) => ({
      type: "result",
      subtype: "success",
      is_error: false,
      result: "ok",
      stop_reason: "end_turn",
      num_turns: 1,
      duration_ms: 1,
      duration_api_ms: 1,
      total_cost_usd: 0,
      usage: {
        input_tokens: usage.input,
        output_tokens: usage.output,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      uuid,
      session_id: "s1",
    });
    const idle = (uuid: string) => ({
      type: "system",
      subtype: "session_state_changed",
      state: "idle",
      uuid,
      session_id: "s1",
    });

    // Prompt A takes the stream (else branch -> promptRunning=true).
    const promptA = agent.prompt({ sessionId: "s1", prompt: [{ type: "text", text: "A" }] });
    await vi.waitFor(() => expect(session.promptRunning).toBe(true));

    // Prompt B arrives while A is running -> queued in pendingMessages.
    const promptB = agent.prompt({ sessionId: "s1", prompt: [{ type: "text", text: "B" }] });
    await vi.waitFor(() => expect(session.pendingMessages.size).toBe(1));

    // A ends at idle without B's user message ever being replayed, so A never
    // hands off; its finally resolves B through the fallback path.
    query.push(result({ input: 0, output: 0 }, "u-a-result"));
    query.push(idle("u-a-idle"));
    await expect(promptA).resolves.toMatchObject({ stopReason: "end_turn" });

    // The fix: B re-asserts promptRunning on resume. Without it this stays
    // false (the fallback cleared it) and the assertion times out.
    await vi.waitFor(() => expect(session.promptRunning).toBe(true));

    // B's own turn is now routed into the turnQueue and consumed, not misrouted
    // off-turn. B resolves with its result and its usage is accounted for.
    query.push(result({ input: 7, output: 3 }, "u-b-result"));
    query.push(idle("u-b-idle"));
    await expect(promptB).resolves.toMatchObject({ stopReason: "end_turn" });
    expect(session.accumulatedUsage.inputTokens).toBe(7);
    expect(session.accumulatedUsage.outputTokens).toBe(3);

    // Only the reader ever consumed the SDK iterator throughout the handoff.
    expect(query.maxConcurrentReads).toBe(1);

    session.abortController.abort();
    query.close();
  });
});
