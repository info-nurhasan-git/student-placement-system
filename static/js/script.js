/* ==========================================================================
   PlacePredict AI - Frontend Logic
   Handles validation, API communication, Chart.js rendering, history and reports.
   ========================================================================== */

(function () {
  "use strict";

  const CONFIG = window.APP_CONFIG || { features: [], featureNames: [] };
  const HISTORY_KEY = "placepredict_history_v1";
  const HISTORY_LIMIT = 5;

  const SAMPLE_VALUES = {
    CGPA: 8.2,
    "10th_Percentage": 51.3,
    "12th_Percentage": 63.8,
    Internships: 1,
    Projects: 1,
    Aptitude_Score: 48,
    Communication_Skills: 9,
    Technical_Skills: 2,
    Backlogs: 3
  };

  const el = {
    form: document.getElementById("predictionForm"),
    predictButton: document.getElementById("predictButton"),
    buttonSpinner: document.querySelector("#predictButton .button-spinner"),
    buttonLabel: document.querySelector("#predictButton .button-label"),
    resetButton: document.getElementById("resetButton"),
    sampleButton: document.getElementById("sampleButton"),
    alertBox: document.getElementById("alertBox"),
    loadingOverlay: document.getElementById("loadingOverlay"),

    resultEmpty: document.getElementById("resultEmpty"),
    resultContent: document.getElementById("resultContent"),
    resultTimestamp: document.getElementById("resultTimestamp"),
    predictionBanner: document.getElementById("predictionBanner"),
    predictionLabel: document.getElementById("predictionLabel"),
    predictionSubtext: document.getElementById("predictionSubtext"),
    predictionIcon: document.getElementById("predictionIcon"),
    probabilityValue: document.getElementById("probabilityValue"),
    probabilityBar: document.getElementById("probabilityBar"),
    chartCenter: document.getElementById("chartCenter"),
    summaryTableBody: document.getElementById("summaryTableBody"),
    disclaimerText: document.getElementById("disclaimerText"),

    downloadButton: document.getElementById("downloadButton"),
    printButton: document.getElementById("printButton"),
    anotherButton: document.getElementById("anotherButton"),

    historyTableBody: document.getElementById("historyTableBody"),
    clearHistoryButton: document.getElementById("clearHistoryButton")
  };

  let probabilityChart = null;
  let latestResult = null;

  /* ----------------------------- Utilities -------------------------------- */

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function showAlert(message, type) {
    if (!el.alertBox) return;
    el.alertBox.className = "alert app-alert alert-" + (type || "danger");
    el.alertBox.textContent = message;
    el.alertBox.classList.remove("d-none");
  }

  function hideAlert() {
    if (!el.alertBox) return;
    el.alertBox.classList.add("d-none");
    el.alertBox.textContent = "";
  }

  function setLoading(isLoading) {
    if (el.loadingOverlay) {
      el.loadingOverlay.classList.toggle("is-visible", isLoading);
      el.loadingOverlay.setAttribute("aria-hidden", String(!isLoading));
    }
    if (el.predictButton) el.predictButton.disabled = isLoading;
    if (el.buttonSpinner) el.buttonSpinner.classList.toggle("d-none", !isLoading);
    if (el.buttonLabel) {
      el.buttonLabel.textContent = isLoading ? "Analysing..." : "Predict Placement";
    }
  }

  function getFeatureLabel(name) {
    const match = CONFIG.features.find(function (feature) {
      return feature.name === name;
    });
    return match ? match.label : String(name).replace(/_/g, " ");
  }

  function readableTime(date) {
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  }

  /* -------------------------- Form validation ----------------------------- */

  function clearFieldError(input) {
    if (!input) return;
    input.classList.remove("is-invalid");
    input.removeAttribute("aria-invalid");
    const feedback = document.querySelector(
      '[data-feedback-for="' + input.name + '"]'
    );
    if (feedback) feedback.textContent = "";
  }

  function setFieldError(input, message) {
    if (!input) return;
    input.classList.add("is-invalid");
    input.setAttribute("aria-invalid", "true");
    const feedback = document.querySelector(
      '[data-feedback-for="' + input.name + '"]'
    );
    if (feedback) feedback.textContent = message;
  }

  function validationMessage(input) {
    const label = input.dataset.label || getFeatureLabel(input.name);
    const value = input.value.trim();

    if (input.validity.valueMissing || value === "") {
      return label + " is required.";
    }
    if (input.validity.badInput) {
      return "Enter a valid number for " + label + ".";
    }
    if (input.validity.rangeUnderflow) {
      return label + " must be at least " + input.min + ".";
    }
    if (input.validity.rangeOverflow) {
      return label + " cannot exceed " + input.max + ".";
    }
    if (input.validity.stepMismatch) {
      return "Use a valid increment (step " + input.step + ") for " + label + ".";
    }
    return "Check the value entered for " + label + ".";
  }

  function validateForm() {
    if (!el.form) return false;
    let isValid = true;
    el.form.classList.add("was-validated");

    CONFIG.featureNames.forEach(function (name) {
      const input = el.form.elements.namedItem(name);
      clearFieldError(input);
      if (!input || !input.checkValidity()) {
        setFieldError(input, input ? validationMessage(input) : "Field not found.");
        isValid = false;
      }
    });

    return isValid;
  }

  function collectPayload() {
    const payload = {};
    CONFIG.featureNames.forEach(function (name) {
      const input = el.form.elements.namedItem(name);
      payload[name] = Number(input.value);
    });
    return payload;
  }

  /* ---------------------------- Chart rendering --------------------------- */

  function renderChart(probabilityPercent, isPlaced) {
    const canvas = document.getElementById("probabilityChart");
    if (!canvas) return;

    const percent = Number(probabilityPercent);
    const primaryColor = isPlaced ? "#10b981" : "#f97316";
    if (el.chartCenter) el.chartCenter.textContent = percent.toFixed(1) + "%";

    if (typeof Chart === "undefined") {
      console.warn("Chart.js is unavailable; showing numeric probability only.");
      return;
    }

    const chartData = {
      labels: ["Placement probability", "Remaining"],
      datasets: [
        {
          data: [percent, Math.max(0, 100 - percent)],
          backgroundColor: [primaryColor, "#e2e8f0"],
          borderWidth: 0,
          hoverOffset: 4
        }
      ]
    };

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "78%",
      animation: { duration: 800 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (context) {
              return context.label + ": " + Number(context.raw).toFixed(2) + "%";
            }
          }
        }
      }
    };

    if (probabilityChart) {
      probabilityChart.data = chartData;
      probabilityChart.options = chartOptions;
      probabilityChart.update();
      return;
    }

    probabilityChart = new Chart(canvas.getContext("2d"), {
      type: "doughnut",
      data: chartData,
      options: chartOptions
    });
  }

  /* --------------------------- Result rendering --------------------------- */

  function renderResult(data, displayTime) {
    latestResult = data;

    const isPlaced = data.prediction === "Placed";
    const percent = Number(data.probability_percent);

    if (el.resultEmpty) el.resultEmpty.classList.add("d-none");
    if (el.resultContent) el.resultContent.classList.remove("d-none");
    if (el.resultTimestamp) el.resultTimestamp.textContent = displayTime;

    if (el.predictionBanner) {
      el.predictionBanner.classList.toggle("is-not-placed", !isPlaced);
    }
    if (el.predictionLabel) el.predictionLabel.textContent = data.prediction;
    if (el.predictionIcon) el.predictionIcon.textContent = isPlaced ? "✓" : "!";
    if (el.predictionSubtext) {
      el.predictionSubtext.textContent = isPlaced
        ? "Model output is at or above the 0.50 decision threshold."
        : "Model output is below the 0.50 decision threshold.";
    }

    if (el.probabilityValue) {
      el.probabilityValue.textContent = percent.toFixed(2) + "%";
    }
    if (el.probabilityBar) {
      el.probabilityBar.style.width = Math.min(100, Math.max(0, percent)) + "%";
    }
    if (el.disclaimerText) {
      el.disclaimerText.textContent = data.disclaimer || "";
    }

    if (el.summaryTableBody && Array.isArray(data.features)) {
      el.summaryTableBody.innerHTML = data.features
        .map(function (feature) {
          return (
            "<tr><td>" +
            escapeHtml(feature.label || getFeatureLabel(feature.name)) +
            "</td><td>" +
            escapeHtml(feature.value) +
            "</td></tr>"
          );
        })
        .join("");
    }

    renderChart(percent, isPlaced);
  }

  function resetResult() {
    latestResult = null;
    if (el.resultContent) el.resultContent.classList.add("d-none");
    if (el.resultEmpty) el.resultEmpty.classList.remove("d-none");
    if (el.resultTimestamp) el.resultTimestamp.textContent = "—";
    if (el.probabilityBar) el.probabilityBar.style.width = "0%";
    if (el.chartCenter) el.chartCenter.textContent = "—";

    if (probabilityChart) {
      probabilityChart.destroy();
      probabilityChart = null;
    }
  }

  /* ------------------------------- History -------------------------------- */

  function readHistory() {
    try {
      const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function writeHistory(history) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, HISTORY_LIMIT)));
    } catch (error) {
      console.warn("Could not save prediction history.", error);
    }
  }

  function addToHistory(payload, data, displayTime) {
    const history = readHistory();
    history.unshift({
      id: Date.now(),
      displayTime: displayTime,
      prediction: data.prediction,
      probability: Number(data.probability_percent),
      values: payload
    });
    writeHistory(history);
    renderHistory();
  }

  function renderHistory() {
    if (!el.historyTableBody) return;
    const history = readHistory();

    if (!history.length) {
      el.historyTableBody.innerHTML =
        '<tr class="history-empty-row"><td colspan="4" class="text-center text-muted py-4">' +
        "No predictions recorded yet.</td></tr>";
      return;
    }

    el.historyTableBody.innerHTML = history
      .map(function (entry, index) {
        const placed = entry.prediction === "Placed";
        const cssClass = placed ? "placed" : "not-placed";
        return (
          "<tr>" +
          "<td>" + escapeHtml(entry.displayTime) + "</td>" +
          '<td><span class="history-pill ' + cssClass + '">' +
          escapeHtml(entry.prediction) + "</span></td>" +
          "<td>" + Number(entry.probability).toFixed(2) + "%</td>" +
          '<td class="text-end">' +
          '<button type="button" class="btn btn-sm btn-outline-primary" data-load-id="' +
          index + '">Load</button>' +
          "</td></tr>"
        );
      })
      .join("");
  }

  function loadFromHistory(index) {
    const history = readHistory();
    const entry = history[index];
    if (!entry || !entry.values || !el.form) return;

    CONFIG.featureNames.forEach(function (name) {
      const input = el.form.elements.namedItem(name);
      if (input && Object.prototype.hasOwnProperty.call(entry.values, name)) {
        input.value = entry.values[name];
        clearFieldError(input);
      }
    });

    document.getElementById("predict")?.scrollIntoView({ behavior: "smooth" });
    showAlert("History entry loaded. Run the prediction again if needed.", "success");
  }

  /* --------------------------- Report download ---------------------------- */

  function downloadReport() {
    if (!latestResult) return;

    const report = {
      generated_at: new Date().toISOString(),
      application: "PlacePredict AI",
      prediction: latestResult.prediction,
      probability: latestResult.probability,
      probability_percent: latestResult.probability_percent,
      decision_threshold: latestResult.threshold,
      student_inputs: latestResult.features,
      model: latestResult.model,
      disclaimer: latestResult.disclaimer
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);

    link.href = url;
    link.download = "placement-report-" + stamp + ".json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  /* ---------------------------- API calls --------------------------------- */

  async function submitPrediction(payload) {
    setLoading(true);
    hideAlert();

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json"
        },
        body: JSON.stringify(payload)
      });

      let data;
      try {
        data = await response.json();
      } catch (error) {
        throw new Error("The server returned an unreadable response.");
      }

      if (!response.ok || !data.success) {
        const message = data.error || "Prediction request failed.";
        if (Array.isArray(data.missing_fields) && data.missing_fields.length) {
          showAlert(message + " Missing: " + data.missing_fields.join(", "));
        } else if (data.invalid_field) {
          showAlert(message + " Field: " + getFeatureLabel(data.invalid_field));
        } else {
          showAlert(message);
        }
        return;
      }

      const now = new Date();
      renderResult(data, readableTime(now));
      addToHistory(payload, data, readableTime(now));
      showAlert("Prediction completed successfully.", "success");
    } catch (error) {
      showAlert(
        "Unable to reach the prediction service. Please check that the Flask server is running."
      );
    } finally {
      setLoading(false);
    }
  }

  /* --------------------------- Event wiring ------------------------------- */

  document.addEventListener("DOMContentLoaded", function () {
    renderHistory();

    if (el.form) {
      el.form.addEventListener("submit", function (event) {
        event.preventDefault();
        if (!validateForm()) {
          showAlert("Please correct the highlighted fields before predicting.");
          return;
        }
        submitPrediction(collectPayload());
      });

      el.form.addEventListener("input", function (event) {
        if (event.target instanceof HTMLInputElement) {
          clearFieldError(event.target);
          hideAlert();
        }
      });

      el.form.addEventListener("reset", function () {
        setTimeout(function () {
          el.form.classList.remove("was-validated");
          CONFIG.featureNames.forEach(function (name) {
            clearFieldError(el.form.elements.namedItem(name));
          });
          resetResult();
          hideAlert();
        }, 0);
      });
    }

    if (el.sampleButton) {
      el.sampleButton.addEventListener("click", function () {
        if (!el.form) return;
        CONFIG.featureNames.forEach(function (name) {
          const input = el.form.elements.namedItem(name);
          if (input && Object.prototype.hasOwnProperty.call(SAMPLE_VALUES, name)) {
            input.value = SAMPLE_VALUES[name];
            clearFieldError(input);
          }
        });
        hideAlert();
      });
    }

    if (el.anotherButton) {
      el.anotherButton.addEventListener("click", function () {
        if (el.form) el.form.reset();
        document.getElementById("predict")?.scrollIntoView({ behavior: "smooth" });
      });
    }

    if (el.downloadButton) {
      el.downloadButton.addEventListener("click", downloadReport);
    }

    if (el.printButton) {
      el.printButton.addEventListener("click", function () {
        window.print();
      });
    }

    if (el.historyTableBody) {
      el.historyTableBody.addEventListener("click", function (event) {
        const button = event.target.closest("[data-load-id]");
        if (button) loadFromHistory(Number(button.dataset.loadId));
      });
    }

    if (el.clearHistoryButton) {
      el.clearHistoryButton.addEventListener("click", function () {
        writeHistory([]);
        renderHistory();
      });
    }
  });
})();
