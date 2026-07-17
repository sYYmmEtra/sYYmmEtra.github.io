import {
  mkdtempSync,
  mkdirSync,
  realpathSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { assertDisjointRoots, assertSafeWritePath } from "../../scripts/lib/paths";

const temporaryRoots: string[] = [];

function makeTemporaryRoot(): string {
  const root = mkdtempSync(
    path.join(realpathSync(tmpdir()), "personal-blog-paths-"),
  );
  temporaryRoots.push(root);
  return root;
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("assertDisjointRoots", () => {
  it("rejects a website nested inside the source root", () => {
    const root = makeTemporaryRoot();
    const sourceRoot = path.join(root, "source");
    const websiteRoot = path.join(sourceRoot, "website");
    mkdirSync(websiteRoot, { recursive: true });

    expect(() => assertDisjointRoots(websiteRoot, sourceRoot)).toThrow(
      /must be physically separate/,
    );
  });

  it("rejects a source root nested inside the website", () => {
    const root = makeTemporaryRoot();
    const websiteRoot = path.join(root, "website");
    const sourceRoot = path.join(websiteRoot, "source");
    mkdirSync(sourceRoot, { recursive: true });

    expect(() => assertDisjointRoots(websiteRoot, sourceRoot)).toThrow(
      /must be physically separate/,
    );
  });

  it("accepts sibling website and source roots", () => {
    const root = makeTemporaryRoot();
    const websiteRoot = path.join(root, "website");
    const sourceRoot = path.join(root, "source");
    mkdirSync(websiteRoot, { recursive: true });
    mkdirSync(sourceRoot, { recursive: true });

    expect(() => assertDisjointRoots(websiteRoot, sourceRoot)).not.toThrow();
  });

  it("rejects identical website and source roots", () => {
    const root = makeTemporaryRoot();
    const sharedRoot = path.join(root, "shared");
    mkdirSync(sharedRoot);

    expect(() => assertDisjointRoots(sharedRoot, sharedRoot)).toThrow(
      /must be physically separate/,
    );
  });
});

describe("assertSafeWritePath", () => {
  it("rejects a write target outside the website root", () => {
    const root = makeTemporaryRoot();
    const websiteRoot = path.join(root, "site");
    const outsideTarget = path.join(root, "site2", "index.html");
    mkdirSync(websiteRoot, { recursive: true });

    expect(() => assertSafeWritePath(websiteRoot, outsideTarget)).toThrow(
      /outside website root/,
    );
  });

  it("rejects a write through a symlink that escapes the website root", () => {
    const root = makeTemporaryRoot();
    const websiteRoot = path.join(root, "site");
    const outsideRoot = path.join(root, "outside");
    const linkedOutside = path.join(websiteRoot, "linked-out");
    mkdirSync(websiteRoot);
    mkdirSync(outsideRoot);
    symlinkSync(outsideRoot, linkedOutside);

    expect(() =>
      assertSafeWritePath(websiteRoot, path.join(linkedOutside, "escape.md")),
    ).toThrow(/outside website root/);
  });

  it("rejects parent traversal after a symlink component", () => {
    const root = makeTemporaryRoot();
    const websiteRoot = path.join(root, "site");
    const outsideRoot = path.join(root, "outside");
    const linkedOutside = path.join(websiteRoot, "linked-out");
    mkdirSync(websiteRoot);
    mkdirSync(outsideRoot);
    symlinkSync(outsideRoot, linkedOutside);
    const target = `${linkedOutside}/../escaped.md`;

    expect(() => assertSafeWritePath(websiteRoot, target)).toThrow(
      /outside website root/,
    );
  });

  it("allows a missing target below a symlinked website root", () => {
    const root = makeTemporaryRoot();
    const physicalWebsiteRoot = path.join(root, "site");
    const websiteRoot = path.join(root, "site-alias");
    mkdirSync(physicalWebsiteRoot);
    symlinkSync(physicalWebsiteRoot, websiteRoot);

    expect(() =>
      assertSafeWritePath(
        websiteRoot,
        path.join(websiteRoot, "generated", "lesson.md"),
      ),
    ).not.toThrow();
  });
});
