import fs from "node:fs";
import path from "node:path";

export function resolveExistingPath(value: string): string {
  return fs.realpathSync(value);
}

function isDescendant(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate);

  return (
    relative !== "" &&
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(relative)
  );
}

export function assertDisjointRoots(
  websiteRoot: string,
  sourceRoot: string,
): void {
  const website = path.resolve(resolveExistingPath(websiteRoot));
  const source = path.resolve(resolveExistingPath(sourceRoot));

  if (
    website === source ||
    isDescendant(website, source) ||
    isDescendant(source, website)
  ) {
    throw new Error("Website and source roots must be physically separate");
  }
}

export function assertSafeWritePath(
  websiteRoot: string,
  target: string,
): void {
  const website = path.resolve(resolveExistingPath(websiteRoot));
  const writeTarget = path.resolve(target);

  if (!isDescendant(website, writeTarget)) {
    throw new Error("Write target is outside website root");
  }
}
