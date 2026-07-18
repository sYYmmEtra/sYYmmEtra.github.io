import { describe, expect, it } from "vitest";

import { runLinkCheck, type LinkChecker } from "../../scripts/check-links";

describe("runLinkCheck", () => {
  it("uses an allocated nonzero port and the production link policy", async () => {
    let receivedOptions: Parameters<LinkChecker>[0] | undefined;
    const messages: string[] = [];

    const result = await runLinkCheck({
      allocatePort: async () => 43127,
      checker: async (options) => {
        receivedOptions = options;
        return { passed: true, links: [] };
      },
      logger: (message) => messages.push(message),
    });

    expect(receivedOptions).toMatchObject({
      path: "dist",
      port: 43127,
      recurse: true,
    });
    expect(receivedOptions?.linksToSkip).toBeTypeOf("function");
    const linksToSkip = receivedOptions?.linksToSkip;
    if (typeof linksToSkip !== "function") {
      throw new Error("Expected functional link policy");
    }
    await expect(linksToSkip("http://localhost:43127/about/")).resolves.toBe(false);
    await expect(linksToSkip("https://www.promptingguide.ai/")).resolves.toBe(true);
    await expect(linksToSkip("mailto:hello@example.com")).resolves.toBe(true);
    expect(receivedOptions?.urlRewriteExpressions).toHaveLength(1);
    expect(receivedOptions?.urlRewriteExpressions?.[0]?.pattern.source).toBe(
      "^https:\\/\\/syymmetra\\.github\\.io\\/",
    );
    expect(receivedOptions?.urlRewriteExpressions?.[0]?.replacement).toBe("http://localhost:43127/");
    expect(result).toEqual({ passed: true, checkedLinks: 0 });
    expect(messages).toEqual(["Checked 0 links."]);
  });

  it("reports broken links and rejects the gate", async () => {
    const checker: LinkChecker = async () => ({
      passed: false,
      links: [{ url: "http://localhost:43127/missing", status: 404, state: "BROKEN" }],
    });

    await expect(runLinkCheck({ checker, allocatePort: async () => 43127 })).rejects.toThrow(
      "Link check failed: 404 http://localhost:43127/missing",
    );
  });
});
