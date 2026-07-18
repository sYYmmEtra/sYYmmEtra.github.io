import { createServer } from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { check, type CheckOptions } from "linkinator";

interface CheckedLink {
  url: string;
  status: number;
  state: string;
}

interface LinkCheckResult {
  passed: boolean;
  links: CheckedLink[];
}

export type LinkChecker = (options: CheckOptions) => Promise<LinkCheckResult>;

export interface RunLinkCheckOptions {
  allocatePort?: () => Promise<number>;
  checker?: LinkChecker;
  logger?: (message: string) => void;
}

export async function allocateLoopbackPort(): Promise<number> {
  const server = createServer();
  try {
    return await new Promise<number>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => {
        const address = server.address();
        if (!address || typeof address === "string") {
          reject(new Error("Could not allocate a loopback port"));
          return;
        }
        resolve(address.port);
      });
    });
  } finally {
    if (server.listening) {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => error ? reject(error) : resolve());
      });
    }
  }
}

export async function runLinkCheck(options: RunLinkCheckOptions = {}): Promise<{
  passed: true;
  checkedLinks: number;
}> {
  const allocatePort = options.allocatePort ?? allocateLoopbackPort;
  const checker = options.checker ?? check;
  const logger = options.logger ?? console.log;
  const port = await allocatePort();
  const result = await checker({
    path: "dist",
    port,
    recurse: true,
    linksToSkip: ["mailto:", "https://github.com/"],
    urlRewriteExpressions: [{
      pattern: /^https:\/\/syymmetra\.github\.io\//,
      replacement: `http://localhost:${port}/`,
    }],
  });

  if (!result.passed) {
    const broken = result.links
      .filter((link) => link.state === "BROKEN")
      .slice(0, 10)
      .map((link) => `${link.status} ${link.url}`)
      .join(", ");
    throw new Error(`Link check failed${broken ? `: ${broken}` : ""}`);
  }

  logger(`Checked ${result.links.length} links.`);
  return { passed: true, checkedLinks: result.links.length };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runLinkCheck().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
