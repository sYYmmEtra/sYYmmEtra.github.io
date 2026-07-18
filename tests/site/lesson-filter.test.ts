import { describe, expect, it, vi } from "vitest";
import {
  createFilterNavigation,
  parseLessonFilters,
  toFilterUrl,
  type LessonFilters,
} from "../../src/scripts/lesson-filter";

const allFilters: LessonFilters = { tracks: ["A"], depths: ["L2"], from: "2026-07-06", to: "2026-07-17" };

describe("lesson filter URL history", () => {
  it("drops invalid filter values while retaining unrelated query parameters", () => {
    const filters = parseLessonFilters("?ref=home&track=A&track=X&depth=L2&depth=L9&from=2026-02-30&to=2026-07-17");

    expect(filters).toEqual({ tracks: ["A"], depths: ["L2"], from: "", to: "2026-07-17" });
    expect(toFilterUrl("https://example.test/ai-daily/?ref=home&track=X#archive", filters)).toBe(
      "/ai-daily/?ref=home&track=A&depth=L2&to=2026-07-17#archive",
    );
  });

  it("pushes user filter changes, canonicalizes initialization, and rehydrates back-forward state without another push", () => {
    const location = {
      href: "https://example.test/ai-daily/?ref=home&track=X",
      pathname: "/ai-daily/",
      search: "?ref=home&track=X",
      hash: "",
    };
    const history = { pushState: vi.fn(), replaceState: vi.fn() };
    const hydrate = vi.fn();
    const navigation = createFilterNavigation(location, history, hydrate);

    navigation.initialize();
    expect(history.replaceState).toHaveBeenCalledWith(null, "", "/ai-daily/?ref=home");
    expect(hydrate).toHaveBeenLastCalledWith({ tracks: [], depths: [], from: "", to: "" });

    navigation.userChange(allFilters);
    expect(history.pushState).toHaveBeenCalledWith(
      null,
      "",
      "/ai-daily/?ref=home&track=A&depth=L2&from=2026-07-06&to=2026-07-17",
    );

    location.search = "?ref=home&track=C";
    navigation.popstate();
    expect(hydrate).toHaveBeenLastCalledWith({ tracks: ["C"], depths: [], from: "", to: "" });
    expect(history.pushState).toHaveBeenCalledTimes(1);
  });
});
