const validTracks = new Set(["A", "B", "C"]);
const validDepths = new Set(["L1", "L2", "L3", "L4"]);
const filterParameterNames = ["track", "depth", "from", "to"] as const;

export interface LessonFilters {
  tracks: string[];
  depths: string[];
  from: string;
  to: string;
}

interface FilterLocation {
  href: string;
  pathname: string;
  search: string;
  hash: string;
}

interface FilterHistory {
  pushState(data: unknown, unused: string, url?: string | URL | null): void;
  replaceState(data: unknown, unused: string, url?: string | URL | null): void;
}

function isValidDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const [year, month, day] = value.split("-").map(Number);
  const parsed = new Date(Date.UTC(year!, month! - 1, day!));
  return parsed.getUTCFullYear() === year && parsed.getUTCMonth() === month! - 1 && parsed.getUTCDate() === day;
}

function uniqueValid(values: readonly string[], allowed: Set<string>): string[] {
  return [...new Set(values.filter((value) => allowed.has(value)))];
}

export function normalizeLessonFilters(filters: LessonFilters): LessonFilters {
  return {
    tracks: uniqueValid(filters.tracks, validTracks),
    depths: uniqueValid(filters.depths, validDepths),
    from: isValidDate(filters.from) ? filters.from : "",
    to: isValidDate(filters.to) ? filters.to : "",
  };
}

export function parseLessonFilters(search: string | URLSearchParams): LessonFilters {
  const params = typeof search === "string" ? new URLSearchParams(search) : search;
  return normalizeLessonFilters({
    tracks: params.getAll("track"),
    depths: params.getAll("depth"),
    from: params.get("from") ?? "",
    to: params.get("to") ?? "",
  });
}

export function toFilterUrl(href: string, filters: LessonFilters): string {
  const url = new URL(href);
  const normalized = normalizeLessonFilters(filters);
  for (const name of filterParameterNames) url.searchParams.delete(name);
  normalized.tracks.forEach((track) => url.searchParams.append("track", track));
  normalized.depths.forEach((depth) => url.searchParams.append("depth", depth));
  if (normalized.from) url.searchParams.set("from", normalized.from);
  if (normalized.to) url.searchParams.set("to", normalized.to);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function createFilterNavigation(
  location: FilterLocation,
  history: FilterHistory,
  hydrate: (filters: LessonFilters) => void,
) {
  const currentPath = () => `${location.pathname}${location.search}${location.hash}`;
  const update = (method: "pushState" | "replaceState", filters: LessonFilters) => {
    const nextUrl = toFilterUrl(location.href, filters);
    if (nextUrl !== currentPath()) history[method](null, "", nextUrl);
  };
  const hydrateCurrent = () => hydrate(parseLessonFilters(location.search));

  return {
    initialize() {
      const filters = parseLessonFilters(location.search);
      update("replaceState", filters);
      hydrate(filters);
    },
    userChange(filters: LessonFilters) {
      const normalized = normalizeLessonFilters(filters);
      update("pushState", normalized);
      hydrate(normalized);
    },
    popstate: hydrateCurrent,
  };
}

function initializeFilters(doc: Document = document, browser: Window = window): void {
  const form = doc.querySelector<HTMLFormElement>("[data-lesson-filters]");
  if (!form) return;

  const cards = [...doc.querySelectorAll<HTMLElement>("[data-lesson-card]")];
  const empty = doc.querySelector<HTMLElement>("[data-lesson-no-results]");
  const status = doc.querySelector<HTMLElement>("[data-lesson-filter-status]");
  const clear = doc.querySelector<HTMLButtonElement>("[data-lesson-clear]");
  const readInputs = (): LessonFilters => ({
    tracks: [...form.querySelectorAll<HTMLInputElement>('input[name="track"]:checked')].map((input) => input.value),
    depths: [...form.querySelectorAll<HTMLInputElement>('input[name="depth"]:checked')].map((input) => input.value),
    from: (form.elements.namedItem("from") as HTMLInputElement | null)?.value ?? "",
    to: (form.elements.namedItem("to") as HTMLInputElement | null)?.value ?? "",
  });
  const setInputs = (filters: LessonFilters) => {
    for (const input of form.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')) {
      const active = input.name === "track" ? filters.tracks : filters.depths;
      input.checked = active.includes(input.value);
    }
    for (const name of ["from", "to"] as const) {
      const input = form.elements.namedItem(name) as HTMLInputElement | null;
      if (input) input.value = filters[name];
    }
  };
  const applyCards = (filters: LessonFilters) => {
    let count = 0;
    for (const card of cards) {
      const match = (!filters.tracks.length || filters.tracks.includes(card.dataset.track ?? ""))
        && (!filters.depths.length || filters.depths.includes(card.dataset.depth ?? ""))
        && (!filters.from || (card.dataset.date ?? "") >= filters.from)
        && (!filters.to || (card.dataset.date ?? "") <= filters.to);
      card.hidden = !match;
      if (match) count += 1;
    }
    if (empty) empty.hidden = count !== 0;
    if (status) status.textContent = `${count} lesson${count === 1 ? "" : "s"}`;
  };
  const navigation = createFilterNavigation(browser.location, browser.history, (filters) => {
    setInputs(filters);
    applyCards(filters);
  });

  navigation.initialize();
  form.addEventListener("change", () => navigation.userChange(readInputs()));
  clear?.addEventListener("click", () => { form.reset(); navigation.userChange(readInputs()); });
  browser.addEventListener("popstate", navigation.popstate);
}

if (typeof document !== "undefined" && typeof window !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => initializeFilters(), { once: true });
  else initializeFilters();
}
