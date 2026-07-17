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

function resolvePhysicalDestination(target: string): string {
  let ancestor = path.resolve(target);
  const missingSuffix: string[] = [];

  while (true) {
    try {
      fs.lstatSync(ancestor);
      return path.resolve(resolveExistingPath(ancestor), ...missingSuffix);
    } catch (error) {
      if (
        !(error instanceof Error) ||
        !("code" in error) ||
        error.code !== "ENOENT"
      ) {
        throw error;
      }

      const parent = path.dirname(ancestor);
      if (parent === ancestor) {
        throw error;
      }

      missingSuffix.unshift(path.basename(ancestor));
      ancestor = parent;
    }
  }
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
  if (target.split(/[\\/]+/).includes("..")) {
    throw new Error("Write target is outside website root");
  }

  const website = path.resolve(resolveExistingPath(websiteRoot));
  const writeTarget = resolvePhysicalDestination(target);

  if (!isDescendant(website, writeTarget)) {
    throw new Error("Write target is outside website root");
  }
}
