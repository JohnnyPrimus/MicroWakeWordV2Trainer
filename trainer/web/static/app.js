"use strict";

const DEFAULT_LANGUAGE = "zh-CN";
const LANGUAGE_STORAGE_KEY = "wakeword-studio-language";

const translations = {
  "zh-CN": {
    "meta.title": "Wakeword Studio · microWakeWord V2",
    "a11y.skipToContent": "跳到主要内容",
    "a11y.javascriptRequired": "此控制台需要启用 JavaScript。",
    "brand.home": "Wakeword Studio 首页",
    "status.connecting": "正在连接",
    "status.ready": "服务就绪",
    "status.running": "任务运行中",
    "status.error": "连接失败",
    "status.configError": "配置错误",
    "nav.apiDocs": "API 文档",
    "language.label": "界面语言",
    "language.switchChinese": "切换为中文",
    "language.switchEnglish": "切换为英文",
    "language.announcement": "界面语言已切换为中文。",
    "hero.eyebrow": "本地训练控制台",
    "hero.title": "从一句话到可部署模型",
    "hero.description":
      "配置唤醒词、生成语音、构建特征并训练 ESPHome 可用的 V2 TFLite 模型。所有任务都在当前容器中运行，结果保存在挂载的本地工作区。",
    "hero.runAll": "运行数据与训练流程",
    "hero.runNext": "执行下一步",
    "hero.summaryLabel": "当前项目摘要",
    "hero.targetWord": "目标唤醒词",
    "hero.runtimeMode": "运行模式",
    "hero.diskFree": "可用磁盘",
    "common.loading": "读取中",
    "common.download": "下载",
    "common.requestFailed": "请求失败（{status}）",
    "config.loadError": "配置无法加载。",
    "workflow.eyebrow": "训练流水线",
    "workflow.title": "训练流程",
    "workflow.description": "按顺序执行。已完成的步骤可以安全重跑，下载与生成的数据会被复用。",
    "state.done": "已完成",
    "state.pending": "待运行",
    "state.optional": "可选",
    "step.setup.title": "安装训练环境",
    "step.setup.description": "克隆 microWakeWord 与 Piper，下载语音模型并安装依赖。首次执行耗时较长。",
    "step.setup.action": "执行",
    "step.previewWord.title": "检查唤醒词发音",
    "step.previewWord.description": "生成一条试听音频，确认 YAML 中的 phonetic 拼写符合预期。",
    "step.previewWord.action": "试听",
    "step.generateSamples.title": "生成训练样本",
    "step.generateSamples.description": "使用 Piper 生成完整的多说话人和多语速样本集。",
    "step.generateSamples.action": "生成",
    "step.downloadData.title": "下载背景与负样本",
    "step.downloadData.description": "下载 RIR、Audioset、FMA 和预生成负特征，通常需要数 GB 空间。",
    "step.downloadData.action": "下载",
    "step.previewAugment.title": "试听增强样本",
    "step.previewAugment.description": "随机混合背景噪声和房间响应，快速检查增强配置是否合理。",
    "step.previewAugment.action": "试听",
    "step.buildFeatures.title": "构建频谱特征",
    "step.buildFeatures.description": "增强正样本并生成训练、验证和测试所需的 mmap 特征。",
    "step.buildFeatures.action": "构建",
    "step.train.title": "训练与量化",
    "step.train.description": "训练混合网络，选择最佳权重并转换为流式量化 TFLite 模型。",
    "step.train.convertOnly": "已有检查点时，仅转换模型",
    "step.train.action": "训练",
    "step.export.title": "导出 ESPHome 模型",
    "step.export.description": "生成可下载的 TFLite 与 V2 manifest。导出参数在右侧设置。",
    "step.export.action": "导出",
    "job.eyebrow": "任务动态",
    "job.title": "运行状态",
    "job.stop": "停止任务",
    "job.empty": "尚未启动任务。首次使用请从“安装训练环境”开始。",
    "job.logLabel": "任务日志",
    "job.starting": "任务正在启动，等待日志…",
    "job.state.running": "运行中",
    "job.state.succeeded": "已完成",
    "job.state.failed": "失败",
    "job.state.cancelled": "已停止",
    "job.toast.succeeded": "{action}已完成",
    "job.toast.failed": "任务执行失败，请查看日志定位原因",
    "job.toast.cancelled": "任务已停止",
    "audio.eyebrow": "试听预览",
    "audio.title": "试听文件",
    "audio.wordPreview": "唤醒词发音",
    "audio.augmentPreview": "增强后样本",
    "audio.empty": "生成预览后，音频播放器会显示在这里。",
    "export.title": "导出参数",
    "export.probability": "概率阈值",
    "export.probabilityHelp": "从训练产出的 ROC 文件中选择合适的 cutoff。",
    "export.tensorArenaHelp": "设备加载失败时再提高。",
    "export.author": "作者",
    "export.website": "网站",
    "artifacts.eyebrow": "输出文件",
    "artifacts.title": "模型文件",
    "artifacts.tflite": "TFLite 模型",
    "artifacts.roc": "ROC 评估结果",
    "artifacts.modelSize": "TFLite 模型 · {size}",
    "config.eyebrow": "配置",
    "config.title": "训练配置",
    "config.expand": "展开编辑 YAML",
    "config.description":
      "修改唤醒词、样本数量、增强参数和训练超参数。保存前会执行语法及字段验证，任务运行时不能编辑。",
    "config.editorLabel": "训练配置 YAML",
    "config.path": "路径：",
    "config.save": "保存配置",
    "config.saving": "正在验证并保存…",
    "config.saved": "配置已保存",
    "config.unsaved": "有未保存的修改",
    "config.saveFailed": "配置未保存，请修正后重试",
    "config.stopConfirm": "停止当前任务？已写入的文件会保留，之后可以安全重跑。",
    "config.stopRequested": "已请求停止任务",
    "footer.localOnly": "数据与模型仅保存在本地工作区。",
    "footer.license": "部分音频数据集仅供个人使用，请在使用前核对许可。",
    "sample.current": "当前工作区已有 {count} 条唤醒词样本。",
    "action.setup": "安装训练环境",
    "action.preview-word": "检查唤醒词发音",
    "action.generate-samples": "生成训练样本",
    "action.download-data": "下载背景与负样本",
    "action.preview-augment": "试听增强样本",
    "action.build-features": "构建频谱特征",
    "action.train": "训练与量化",
    "action.export": "导出 ESPHome 模型",
    "action.all": "运行数据与训练流程",
    "toast.started": "已启动：{action}",
  },
  en: {
    "meta.title": "Wakeword Studio · microWakeWord V2",
    "a11y.skipToContent": "Skip to main content",
    "a11y.javascriptRequired": "This control panel requires JavaScript.",
    "brand.home": "Wakeword Studio home",
    "status.connecting": "Connecting",
    "status.ready": "Service ready",
    "status.running": "Job running",
    "status.error": "Connection failed",
    "status.configError": "Configuration error",
    "nav.apiDocs": "API docs",
    "language.label": "Interface language",
    "language.switchChinese": "Switch to Chinese",
    "language.switchEnglish": "Switch to English",
    "language.announcement": "Interface language changed to English.",
    "hero.eyebrow": "Local training console",
    "hero.title": "From a spoken phrase to a deployable model",
    "hero.description":
      "Configure a wake word, generate speech, build features, and train a V2 TFLite model for ESPHome. Every job runs in this container, with results saved to the mounted local workspace.",
    "hero.runAll": "Run data and training pipeline",
    "hero.runNext": "Run next step",
    "hero.summaryLabel": "Current project summary",
    "hero.targetWord": "Target wake word",
    "hero.runtimeMode": "Runtime",
    "hero.diskFree": "Disk available",
    "common.loading": "Loading",
    "common.download": "Download",
    "common.requestFailed": "Request failed ({status})",
    "config.loadError": "The configuration could not be loaded.",
    "workflow.eyebrow": "Training pipeline",
    "workflow.title": "Pipeline",
    "workflow.description":
      "Run the steps in order. Completed steps can be safely re-run, while downloaded and generated data is reused.",
    "state.done": "Complete",
    "state.pending": "Not run",
    "state.optional": "Optional",
    "step.setup.title": "Install training environment",
    "step.setup.description":
      "Clone microWakeWord and Piper, download the voice model, and install dependencies. The first run takes longer.",
    "step.setup.action": "Run",
    "step.previewWord.title": "Check wake word pronunciation",
    "step.previewWord.description":
      "Generate one preview clip and confirm the phonetic spelling in the YAML file.",
    "step.previewWord.action": "Preview",
    "step.generateSamples.title": "Generate training samples",
    "step.generateSamples.description":
      "Use Piper to generate a complete set of multi-speaker and multi-speed samples.",
    "step.generateSamples.action": "Generate",
    "step.downloadData.title": "Download background and negative audio",
    "step.downloadData.description":
      "Download RIR, Audioset, FMA, and prebuilt negative features. Several GB of space is usually required.",
    "step.downloadData.action": "Download",
    "step.previewAugment.title": "Preview augmented sample",
    "step.previewAugment.description":
      "Quickly check the augmentation settings with background noise and room response mixed in.",
    "step.previewAugment.action": "Preview",
    "step.buildFeatures.title": "Build spectrogram features",
    "step.buildFeatures.description":
      "Augment positive samples and build the mmap features used for training, validation, and testing.",
    "step.buildFeatures.action": "Build",
    "step.train.title": "Train and quantize",
    "step.train.description":
      "Train the mixed network, select the best weights, and convert them to a streaming quantized TFLite model.",
    "step.train.convertOnly": "With an existing checkpoint, convert only",
    "step.train.action": "Train",
    "step.export.title": "Export ESPHome model",
    "step.export.description":
      "Generate the downloadable TFLite model and V2 manifest. Export settings are on the right.",
    "step.export.action": "Export",
    "job.eyebrow": "Activity",
    "job.title": "Run status",
    "job.stop": "Stop job",
    "job.empty": "No job has started yet. Begin with “Install training environment”.",
    "job.logLabel": "Job log",
    "job.starting": "Starting the job. Waiting for logs…",
    "job.state.running": "Running",
    "job.state.succeeded": "Complete",
    "job.state.failed": "Failed",
    "job.state.cancelled": "Stopped",
    "job.toast.succeeded": "{action} complete",
    "job.toast.failed": "The job failed. Check the log to identify the cause.",
    "job.toast.cancelled": "Job stopped",
    "audio.eyebrow": "Preview",
    "audio.title": "Preview audio",
    "audio.wordPreview": "Wake word pronunciation",
    "audio.augmentPreview": "Augmented sample",
    "audio.empty": "Audio players appear here after a preview is generated.",
    "export.title": "Export settings",
    "export.probability": "Probability cutoff",
    "export.probabilityHelp": "Choose an appropriate cutoff from the ROC output generated during training.",
    "export.tensorArenaHelp": "Increase this if the device fails to load the model.",
    "export.author": "Author",
    "export.website": "Website",
    "artifacts.eyebrow": "Artifacts",
    "artifacts.title": "Model files",
    "artifacts.tflite": "TFLite model",
    "artifacts.roc": "ROC evaluation",
    "artifacts.modelSize": "TFLite model · {size}",
    "config.eyebrow": "Configuration",
    "config.title": "Training configuration",
    "config.expand": "Expand to edit YAML",
    "config.description":
      "Edit the wake word, sample count, augmentation settings, and training hyperparameters. Syntax and fields are validated before saving. Editing is disabled while a job is running.",
    "config.editorLabel": "Training configuration YAML",
    "config.path": "Path: ",
    "config.save": "Save configuration",
    "config.saving": "Validating and saving…",
    "config.saved": "Configuration saved",
    "config.unsaved": "Unsaved changes",
    "config.saveFailed": "Configuration was not saved. Fix the errors and try again.",
    "config.stopConfirm":
      "Stop the current job? Files already written are preserved, and the pipeline can be safely resumed later.",
    "config.stopRequested": "Stop request sent",
    "footer.localOnly": "Data and models remain in the local workspace.",
    "footer.license":
      "Some audio datasets are for personal use only. Check each license before use.",
    "sample.current": "The current workspace contains {count} wake word samples.",
    "action.setup": "Install training environment",
    "action.preview-word": "Check wake word pronunciation",
    "action.generate-samples": "Generate training samples",
    "action.download-data": "Download background and negative audio",
    "action.preview-augment": "Preview augmented sample",
    "action.build-features": "Build spectrogram features",
    "action.train": "Train and quantize",
    "action.export": "Export ESPHome model",
    "action.all": "Run data and training pipeline",
    "toast.started": "Started: {action}",
  },
};

function readStoredLanguage() {
  try {
    return window.localStorage.getItem(LANGUAGE_STORAGE_KEY) === "en" ? "en" : DEFAULT_LANGUAGE;
  } catch {
    return DEFAULT_LANGUAGE;
  }
}

let currentLanguage = readStoredLanguage();

function t(key, values = {}) {
  const template =
    translations[currentLanguage]?.[key] ?? translations[DEFAULT_LANGUAGE][key] ?? key;
  return template.replace(/\{(\w+)\}/g, (match, name) =>
    Object.hasOwn(values, name) ? String(values[name]) : match,
  );
}

function actionLabel(action) {
  const key = `action.${action}`;
  const label = t(key);
  return label === key ? action : label;
}

function jobStateLabel(status) {
  const key = `job.state.${status}`;
  const label = t(key);
  return label === key ? status : label;
}

const stepOrder = [
  "setup",
  "preview-word",
  "generate-samples",
  "download-data",
  "preview-augment",
  "build-features",
  "train",
  "export",
];

const state = {
  status: null,
  jobRunning: false,
  configDirty: false,
  configPath: null,
  configStatus: null,
  serverStatus: { key: "status.connecting", state: "loading" },
  lastFinishedJob: null,
  toastTimer: null,
  polling: false,
};

const elements = {
  serverStatus: document.querySelector("#server-status"),
  serverStatusText: document.querySelector("#server-status .status-text"),
  targetWord: document.querySelector("#target-word"),
  runtimeMode: document.querySelector("#runtime-mode"),
  diskFree: document.querySelector("#disk-free"),
  configError: document.querySelector("#config-error"),
  configErrorMessage: document.querySelector("#config-error-message"),
  runAll: document.querySelector("#run-all-button"),
  runNext: document.querySelector("#run-next-button"),
  cancel: document.querySelector("#cancel-button"),
  jobEmpty: document.querySelector("#job-empty"),
  jobActive: document.querySelector("#job-active"),
  jobAction: document.querySelector("#job-action"),
  jobState: document.querySelector("#job-state"),
  jobLog: document.querySelector("#job-log"),
  wordAudioCard: document.querySelector("#word-audio-card"),
  wordAudio: document.querySelector("#word-audio"),
  augmentAudioCard: document.querySelector("#augment-audio-card"),
  augmentAudio: document.querySelector("#augment-audio"),
  audioEmpty: document.querySelector("#audio-empty"),
  modelPanel: document.querySelector("#model-panel"),
  modelDownload: document.querySelector("#model-download"),
  manifestDownload: document.querySelector("#manifest-download"),
  rocDownload: document.querySelector("#roc-download"),
  sampleDescription: document.querySelector("#sample-description"),
  configEditor: document.querySelector("#config-editor"),
  configPath: document.querySelector("#config-path"),
  configSaveStatus: document.querySelector("#config-save-status"),
  saveConfig: document.querySelector("#save-config-button"),
  toast: document.querySelector("#toast"),
  probabilityCutoff: document.querySelector("#probability-cutoff"),
  tensorArenaSize: document.querySelector("#tensor-arena-size"),
  exportAuthor: document.querySelector("#export-author"),
  exportWebsite: document.querySelector("#export-website"),
  languageOptions: document.querySelectorAll(".language-option"),
  languageAnnouncement: document.querySelector("#language-announcement"),
};

function updateLanguageControls() {
  document.documentElement.lang = currentLanguage;
  document.title = t("meta.title");
  elements.languageOptions.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.language === currentLanguage));
  });
}

function renderConfigStatus() {
  if (!state.configStatus) {
    elements.configSaveStatus.textContent = "";
    elements.configSaveStatus.dataset.state = "";
    return;
  }
  const { key, raw, state: statusState = "" } = state.configStatus;
  elements.configSaveStatus.textContent = key ? t(key) : raw;
  elements.configSaveStatus.dataset.state = statusState;
}

function setConfigStatus({ key = null, raw = "", state: statusState = "" }) {
  state.configStatus = { key, raw, state: statusState };
  renderConfigStatus();
}

function applyTranslations({ renderCurrent = true } = {}) {
  updateLanguageControls();
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach((element) => {
    element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
  });

  elements.configPath.textContent = state.configPath || t("common.loading");
  renderConfigStatus();
  setServerStatus(state.serverStatus.key, state.serverStatus.state);

  if (renderCurrent && state.status) {
    renderStatus(state.status);
  }
}

function setLanguage(language, { persist = true, announce = true } = {}) {
  currentLanguage = language === "en" ? "en" : DEFAULT_LANGUAGE;
  if (persist) {
    try {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLanguage);
    } catch {
      // The selected language still applies for this session.
    }
  }
  applyTranslations();
  elements.languageAnnouncement.textContent = announce ? t("language.announcement") : "";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  let body = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    body = await response.json();
  } else {
    body = await response.text();
  }

  if (!response.ok) {
    const detail = body && typeof body === "object" ? body.detail : body;
    throw new Error(
      typeof detail === "string" ? detail : t("common.requestFailed", { status: response.status }),
    );
  }
  return body;
}

function showToast(message, type = "success") {
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.dataset.state = type;
  elements.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 4200);
}

function setServerStatus(key, status) {
  state.serverStatus = { key, state: status };
  elements.serverStatus.dataset.state = status;
  elements.serverStatusText.textContent = t(key);
  elements.serverStatus.setAttribute("aria-label", t(key));
}

function allActionButtons() {
  return document.querySelectorAll("[data-action]");
}

function updateButtonState() {
  allActionButtons().forEach((button) => {
    button.disabled = state.jobRunning;
  });
  elements.saveConfig.disabled = state.jobRunning;
  elements.runNext.disabled = state.jobRunning;
}

function renderSteps(steps) {
  stepOrder.forEach((stepId) => {
    const card = document.querySelector(`[data-step="${stepId}"]`);
    if (!card) return;
    const step = steps[stepId] || { done: false };
    card.dataset.done = String(Boolean(step.done));
    const badge = card.querySelector(".step-state");
    badge.textContent = step.done
      ? t("state.done")
      : stepId === "preview-augment"
        ? t("state.optional")
        : t("state.pending");
  });

  const sampleCount = steps["generate-samples"]?.sample_count || 0;
  elements.sampleDescription.textContent = sampleCount
    ? t("sample.current", { count: sampleCount.toLocaleString(currentLanguage) })
    : t("step.generateSamples.description");
}

function renderJob(job) {
  const running = Boolean(job && job.status === "running");
  state.jobRunning = running;
  updateButtonState();

  if (!job) {
    elements.jobEmpty.hidden = false;
    elements.jobActive.hidden = true;
    elements.cancel.hidden = true;
    return;
  }

  elements.jobEmpty.hidden = true;
  elements.jobActive.hidden = false;
  elements.cancel.hidden = !running;
  elements.jobAction.textContent = actionLabel(job.action);
  elements.jobState.textContent = jobStateLabel(job.status);
  elements.jobState.dataset.state = job.status;

  const nearBottom =
    elements.jobLog.scrollHeight - elements.jobLog.scrollTop - elements.jobLog.clientHeight < 80;
  if (elements.jobLog.textContent !== job.log) {
    elements.jobLog.textContent = job.log || t("job.starting");
    if (nearBottom || running) {
      elements.jobLog.scrollTop = elements.jobLog.scrollHeight;
    }
  }

  const finishedKey = `${job.id}:${job.status}`;
  if (!running && state.lastFinishedJob !== finishedKey) {
    state.lastFinishedJob = finishedKey;
    if (job.status === "succeeded") {
      showToast(t("job.toast.succeeded", { action: actionLabel(job.action) }));
    } else if (job.status === "failed") {
      showToast(t("job.toast.failed"), "error");
    } else if (job.status === "cancelled") {
      showToast(t("job.toast.cancelled"), "error");
    }
  }
}

function setAudio(card, player, artifact) {
  if (!artifact) {
    card.hidden = true;
    player.removeAttribute("src");
    return false;
  }
  if (player.getAttribute("src") !== artifact.url) {
    player.src = artifact.url;
  }
  card.hidden = false;
  return true;
}

function renderArtifacts(artifacts = {}) {
  const hasWord = setAudio(elements.wordAudioCard, elements.wordAudio, artifacts["word-preview"]);
  const hasAugment = setAudio(
    elements.augmentAudioCard,
    elements.augmentAudio,
    artifacts["augmentation-preview"],
  );
  elements.audioEmpty.hidden = hasWord || hasAugment;

  const hasModel = Boolean(artifacts.model);
  elements.modelPanel.hidden = !hasModel && !artifacts.manifest;
  if (hasModel) {
    elements.modelDownload.hidden = false;
    elements.modelDownload.href = artifacts.model.url;
    elements.modelDownload.querySelector("span:first-child").textContent =
      t("artifacts.modelSize", { size: formatBytes(artifacts.model.size) });
  } else {
    elements.modelDownload.hidden = true;
  }
  if (artifacts.manifest) {
    elements.manifestDownload.hidden = false;
    elements.manifestDownload.href = artifacts.manifest.url;
  } else {
    elements.manifestDownload.hidden = true;
  }
  if (artifacts.roc) {
    elements.rocDownload.hidden = false;
    elements.rocDownload.href = artifacts.roc.url;
  } else {
    elements.rocDownload.hidden = true;
  }
}

function formatBytes(bytes) {
  if (!bytes) return "0 KB";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function renderStatus(data) {
  state.status = data;
  renderJob(data.job);
  setServerStatus(
    data.job?.status === "running" ? "status.running" : "status.ready",
    data.job?.status === "running" ? "running" : "ready",
  );

  if (data.config_error) {
    elements.configError.hidden = false;
    elements.configErrorMessage.textContent = data.config_error;
    elements.targetWord.textContent = t("status.configError");
    setServerStatus("status.configError", "error");
    return;
  }

  elements.configError.hidden = true;
  elements.targetWord.textContent = data.wake_word.phonetic;
  elements.runtimeMode.textContent = data.runtime.mode;
  elements.diskFree.textContent = `${data.disk_free_gb.toLocaleString(currentLanguage)} GB`;
  renderSteps(data.steps || {});
  renderArtifacts(data.artifacts);
}

async function refreshStatus() {
  if (state.polling) return;
  state.polling = true;
  try {
    const data = await api("/api/status");
    renderStatus(data);
  } catch (error) {
    setServerStatus("status.error", "error");
    if (!state.status) {
      elements.configErrorMessage.textContent = error.message;
      elements.configError.hidden = false;
    }
  } finally {
    state.polling = false;
  }
}

async function loadConfig() {
  try {
    const data = await api("/api/config");
    elements.configEditor.value = data.content;
    state.configPath = data.path;
    elements.configPath.textContent = state.configPath;
    state.configDirty = false;
    setConfigStatus({});
  } catch (error) {
    setConfigStatus({ raw: error.message, state: "error" });
  }
}

async function startAction(action, overrides = {}) {
  if (state.jobRunning) return;

  const payload = {
    action,
    probability_cutoff: Number(elements.probabilityCutoff.value || 0.5),
    tensor_arena_size: Number(elements.tensorArenaSize.value || 30000),
    author: elements.exportAuthor.value.trim(),
    website: elements.exportWebsite.value.trim(),
    ...overrides,
  };

  state.jobRunning = true;
  updateButtonState();
  try {
    await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    showToast(t("toast.started", { action: actionLabel(action) }));
    await refreshStatus();
  } catch (error) {
    state.jobRunning = false;
    updateButtonState();
    showToast(error.message, "error");
  }
}

function findNextAction(status) {
  const steps = status?.steps || {};
  if (!steps.setup?.done) return "setup";
  if (!steps["preview-word"]?.done) return "preview-word";
  if (!steps["generate-samples"]?.done) return "generate-samples";
  if (!steps["download-data"]?.done) return "download-data";
  if (!steps["build-features"]?.done) return "build-features";
  if (!steps.train?.done) return "train";
  if (!steps.export?.done) return "export";
  return "export";
}

async function saveConfig() {
  elements.saveConfig.disabled = true;
  setConfigStatus({ key: "config.saving" });
  try {
    await api("/api/config", {
      method: "PUT",
      body: JSON.stringify({ content: elements.configEditor.value }),
    });
    state.configDirty = false;
    setConfigStatus({ key: "config.saved", state: "success" });
    showToast(t("config.saved"));
    await refreshStatus();
  } catch (error) {
    setConfigStatus({ raw: error.message, state: "error" });
    showToast(t("config.saveFailed"), "error");
  } finally {
    elements.saveConfig.disabled = state.jobRunning;
  }
}

document.addEventListener("click", (event) => {
  const languageButton = event.target.closest(".language-option");
  if (languageButton) {
    setLanguage(languageButton.dataset.language);
    return;
  }

  const button = event.target.closest("[data-action]");
  if (!button) return;
  startAction(button.dataset.action, {
    no_train: button.dataset.noTrain === "true",
  });
});

elements.runNext.addEventListener("click", () => {
  if (!state.status) return;
  startAction(findNextAction(state.status));
});

elements.cancel.addEventListener("click", async () => {
  const job = state.status?.job;
  if (!job) return;
  if (!window.confirm(t("config.stopConfirm"))) return;
  try {
    await api(`/api/jobs/${job.id}/cancel`, { method: "POST" });
    showToast(t("config.stopRequested"));
    await refreshStatus();
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.saveConfig.addEventListener("click", saveConfig);

elements.configEditor.addEventListener("input", () => {
  state.configDirty = true;
  setConfigStatus({ key: "config.unsaved" });
});

window.addEventListener("beforeunload", (event) => {
  if (!state.configDirty) return;
  event.preventDefault();
  event.returnValue = "";
});

applyTranslations({ renderCurrent: false });

Promise.all([refreshStatus(), loadConfig()]).then(() => {
  updateButtonState();
  window.setInterval(refreshStatus, 3000);
});
