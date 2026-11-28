export type BrowserModel = {
  provider: "Google Gemini" | "Groq" | "Mistral";
  model: string;
  key: string;
};

type Completion = {
  system: string;
  user: string;
  schema: Record<string, unknown>;
};

class InvalidProviderJsonError extends Error {}

function geminiSchema(schema: Record<string, unknown>): Record<string, unknown> {
  // Pydantic represents an arbitrary nullable value as anyOf: [{}, {type: "null"}].
  // The rubric uses scalar values, so give Gemini an explicit, constrained union.
  return JSON.parse(JSON.stringify(schema, (key, value) => {
    if (key === "default") return undefined;
    if (key === "anyOf" && Array.isArray(value) &&
        value.some((branch) => branch && Object.keys(branch).length === 0)) {
      return ["string", "number", "boolean", "null"].map((type) => ({ type }));
    }
    return value;
  }));
}

async function complete(config: BrowserModel, message: Completion, signal: AbortSignal): Promise<unknown> {
  const { provider, model, key } = config;
  const system = `${message.system}\nReturn one JSON object matching this schema: ${JSON.stringify(message.schema)}`;
  const gemini = provider === "Google Gemini";
  const url = gemini
    ? `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`
    : provider === "Groq"
      ? "https://api.groq.com/openai/v1/chat/completions"
      : "https://api.mistral.ai/v1/chat/completions";
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(gemini ? { "x-goog-api-key": key } : { Authorization: `Bearer ${key}` }),
    },
    credentials: "omit",
    redirect: "error",
    signal,
    body: JSON.stringify(gemini
      ? {
          systemInstruction: { parts: [{ text: system }] },
          contents: [{ role: "user", parts: [{ text: message.user }] }],
          generationConfig: {
            temperature: 0,
            responseMimeType: "application/json",
            responseJsonSchema: geminiSchema(message.schema),
          },
        }
      : {
          model,
          messages: [{ role: "system", content: system }, { role: "user", content: message.user }],
          temperature: 0,
          max_tokens: 2048,
          response_format: { type: "json_object" },
        }),
  });
  if (!response.ok) {
    const status = response.status;
    const retry = response.headers.get("retry-after");
    const hint = status === 429
      ? retry && /^\d+$/.test(retry) ? `Wait ${retry} seconds and try again.` : "Check the provider quota before retrying."
      : status === 401 || status === 403 ? "Check your API key and its model permissions."
      : status === 404 ? "Choose a model available to your account."
      : "Try again or choose another model.";
    throw new Error(`${provider} returned HTTP ${status}. ${hint}`);
  }
  const result = await response.json();
  const text = gemini
    ? result.candidates?.[0]?.content?.parts?.filter((part: { thought?: boolean }) => !part.thought)
        .map((part: { text?: string }) => part.text ?? "").join("")
    : result.choices?.[0]?.message?.content;
  if (typeof text !== "string" || !text.trim()) throw new InvalidProviderJsonError(`${provider} returned no JSON response.`);
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new InvalidProviderJsonError(`${provider} returned invalid JSON.`);
  }
  if (JSON.stringify(value).includes(key)) throw new Error("The provider response contained a credential.");
  return value;
}

export async function testBrowserModel(config: BrowserModel): Promise<void> {
  try {
    const result = await complete(config, {
      system: "Reply with exactly one object.",
      user: 'Return {"ok":true}.',
      schema: { type: "object", properties: { ok: { type: "boolean" } }, required: ["ok"] },
    }, AbortSignal.timeout(20000));
    if (!result || typeof result !== "object" || !("ok" in result) || result.ok !== true) {
      throw new Error("The model connected but did not return the expected test response.");
    }
  } catch (error) {
    if (error instanceof TypeError) throw new Error(`The browser could not reach ${config.provider}. Check your network and whether the provider allows direct browser requests.`);
    if (error instanceof DOMException && error.name === "TimeoutError") throw new Error("The provider connection test timed out.");
    throw error;
  }
}

export function runBrowserModel<T>(
  apiBase: string,
  workspace: string,
  action: string,
  params: object,
  config: BrowserModel,
  onProgress?: (completed: number, total: number) => void,
): Promise<T> {
  const url = new URL(`${apiBase}/api/browser-model`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return new Promise<T>((resolve, reject) => {
    const socket = new WebSocket(url);
    const controller = new AbortController();
    let failure: Error | undefined;
    socket.onopen = () => socket.send(JSON.stringify({ action, params, workspace, model: config.model, provider: config.provider }));
    socket.onmessage = async ({ data }) => {
      const event = JSON.parse(data);
      if (event.type === "result") {
        resolve(event.value as T);
        socket.close();
      } else if (event.type === "error") {
        reject(failure ?? new Error(event.message));
        socket.close();
      } else if (event.type === "progress") {
        onProgress?.(event.completed, event.total);
      } else if (event.type === "completion") {
        try {
          failure = undefined;
          const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(120000)]);
          const value = await complete(config, event, signal);
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "completion_result", value }));
          }
        } catch (error) {
          failure = error instanceof TypeError
            ? new Error(`Could not reach ${config.provider} directly from this browser. Check network access and browser support.`)
            : error instanceof DOMException && error.name === "TimeoutError"
              ? new Error(`${config.provider} did not answer within two minutes.`)
              : error instanceof Error ? error : new Error("Provider request failed.");
          if (socket.readyState === WebSocket.OPEN) {
            const errorKind = error instanceof InvalidProviderJsonError ? "invalid_json" : "provider";
            socket.send(JSON.stringify({ type: "completion_result", error: errorKind }));
          }
        }
      }
    };
    socket.onclose = () => {
      controller.abort();
      reject(failure ?? new Error("The browser model connection closed before the run completed."));
    };
  });
}
