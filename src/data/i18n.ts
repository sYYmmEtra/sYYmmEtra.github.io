type LocalizedShape<T> = {
  [Key in keyof T]: T[Key] extends Record<string, unknown>
    ? LocalizedShape<T[Key]>
    : string;
};

export const en = {
  navigation: {
    home: "Home",
    archive: "AI Daily",
    projects: "Projects",
    about: "About",
    skipToContent: "Skip to content",
  },
  hero: {
    eyebrow: "Software engineer · AI engineering learner",
    title: "Learning in public, building with care.",
    description: "Notes from a deliberate practice in AI engineering.",
    archiveAction: "Browse AI Daily",
    aboutAction: "About me",
  },
  archive: {
    title: "AI Daily",
    description: "A growing archive of learning notes. Original articles are in Chinese.",
    searchLabel: "Search lessons",
    searchPlaceholder: "Search AI Daily",
    filterTrack: "Track",
    filterDepth: "Depth",
    filterDate: "Date",
    clearFilters: "Clear filters",
    originalChinese: "Original in Chinese",
    noResults: "No lessons match these filters.",
  },
  projects: {
    title: "Projects",
    emptyTitle: "Selected case studies are being prepared.",
    emptyDescription: "For current work and experiments, visit GitHub.",
    githubAction: "View GitHub",
  },
  about: {
    title: "About",
    role: "Software engineer learning and practicing AI engineering.",
    learning: "AI Daily is a public record of questions, notes, and practical study.",
    contact: "Get in touch",
  },
  disclosures: {
    personalNote: "Personal learning note",
    aiAssisted: "AI-assisted",
    verifySources: "Verify important sources",
    originalLanguage: "Article body in Chinese",
  },
  sourceStatus: {
    label: "Source status",
    unreviewed: "Unreviewed",
    partiallyVerified: "Partially verified",
    verified: "Verified",
  },
  theme: {
    label: "Theme",
    light: "Use light theme",
    dark: "Use dark theme",
    system: "Use system theme",
    currentLight: "Light theme",
    currentDark: "Dark theme",
    currentSystem: "System theme",
  },
} as const;

export const zhCN = {
  navigation: {
    home: "首页",
    archive: "AI Daily",
    projects: "项目",
    about: "关于",
    skipToContent: "跳至主要内容",
  },
  hero: {
    eyebrow: "软件工程师 · AI 工程学习者",
    title: "公开学习，认真构建。",
    description: "记录 AI 工程实践中的系统学习。",
    archiveAction: "浏览 AI Daily",
    aboutAction: "关于我",
  },
  archive: {
    title: "AI Daily",
    description: "持续更新的学习笔记归档，文章原文为中文。",
    searchLabel: "搜索课程",
    searchPlaceholder: "搜索 AI Daily",
    filterTrack: "轨道",
    filterDepth: "深度",
    filterDate: "日期",
    clearFilters: "清除筛选",
    originalChinese: "中文原文",
    noResults: "没有符合这些筛选条件的课程。",
  },
  projects: {
    title: "项目",
    emptyTitle: "精选案例正在整理中。",
    emptyDescription: "当前工作与实验请访问 GitHub。",
    githubAction: "查看 GitHub",
  },
  about: {
    title: "关于",
    role: "软件工程师，正在学习和实践 AI 工程。",
    learning: "AI Daily 是公开记录问题、笔记与实践学习的地方。",
    contact: "联系我",
  },
  disclosures: {
    personalNote: "个人学习笔记",
    aiAssisted: "AI 辅助",
    verifySources: "重要来源请自行核验",
    originalLanguage: "文章正文为中文",
  },
  sourceStatus: {
    label: "来源状态",
    unreviewed: "未审阅",
    partiallyVerified: "部分核验",
    verified: "已核验",
  },
  theme: {
    label: "主题",
    light: "使用浅色主题",
    dark: "使用深色主题",
    system: "跟随系统主题",
    currentLight: "浅色主题",
    currentDark: "深色主题",
    currentSystem: "系统主题",
  },
} as const satisfies LocalizedShape<typeof en>;

export const dictionaries = { en, "zh-CN": zhCN } as const;
export type Locale = keyof typeof dictionaries;
export type Dictionary = (typeof dictionaries)[Locale];
