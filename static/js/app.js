document.addEventListener("DOMContentLoaded", () => {
    const stage1 = document.getElementById("stage-1");
    const stage2 = document.getElementById("stage-2");
    const stage3 = document.getElementById("stage-3");

    const btnLogin = document.getElementById("btn-login");
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadLabel = document.getElementById("upload-label");
    const btnStartScan = document.getElementById("btn-start-scan");
    
    const epsSlider = document.getElementById("epsilon-slider");
    const epsDisplay = document.getElementById("eps-display");
    const btnGenerateShield = document.getElementById("btn-generate-shield");
    
    const heatmapImg = document.getElementById("heatmap-img");
    const landmarkCountLabel = document.getElementById("landmark-count-label");
    
    const verifyOrigImg = document.getElementById("verify-orig-img");
    const verifyShieldImg = document.getElementById("verify-shield-img");
    const ssimDisplay = document.getElementById("ssim-display");
    const protectionScoreDisplay = document.getElementById("protection-score-display");
    const btnExportPurge = document.getElementById("btn-export-purge");
    const purgeCheckbox = document.getElementById("purge-checkbox");
    const btnReset = document.getElementById("btn-reset");

    let currentFile = null;
    let confidenceChart = null;

    function showStage(stage) {
        [stage1, stage2, stage3].forEach(s => {
            s.classList.remove("active");
            s.classList.add("hidden");
        });
        stage.classList.remove("hidden");
        stage.classList.add("active");
    }

    btnLogin.addEventListener("click", async () => {
        btnLogin.innerText = "AUTHENTICATING...";
        const res = await fetch("/api/auth", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username: "researcher@vtu.ac.in", password: "session_password" })
        });
        const data = await res.json();
        if (data.status === "success") {
            btnLogin.innerText = "AUTHENTICATED (LOCAL ACTIVE)";
            btnLogin.classList.replace("bg-indigo-600", "bg-emerald-600");
        }
    });

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => e.preventDefault());
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        currentFile = file;
        uploadLabel.innerText = file.name.substring(0, 24);
        btnStartScan.disabled = false;
    }

    btnStartScan.addEventListener("click", async () => {
        if (!currentFile) return;
        btnStartScan.innerText = "SCANNING...";
        const formData = new FormData();
        formData.append("file", currentFile);

        const res = await fetch("/api/scan_vulnerability", { method: "POST", body: formData });
        const data = await res.json();
        if (data.status === "success") {
            heatmapImg.src = data.heatmap_preview;
            verifyOrigImg.src = data.original_preview;
            landmarkCountLabel.innerText = `Isolating ${data.landmarks_count} identity-bearing facial landmarks`;
            showStage(stage2);
        }
        btnStartScan.innerText = "START SCAN";
    });

    epsSlider.addEventListener("input", (e) => {
        epsDisplay.innerText = parseFloat(e.target.value).toFixed(3);
    });

    btnGenerateShield.addEventListener("click", async () => {
        if (!currentFile) return;
        btnGenerateShield.innerText = "OPTIMIZING PGD TENSOR...";
        btnGenerateShield.disabled = true;

        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("epsilon", epsSlider.value);
        formData.append("iterations", 10);

        const res = await fetch("/api/shield", { method: "POST", body: formData });
        const data = await res.json();

        if (data.status === "success") {
            verifyShieldImg.src = data.shielded_preview;
            ssimDisplay.innerText = data.ssim;
            protectionScoreDisplay.innerText = `${data.protection_score}% (Approved)`;
            renderChart(data.baseline_confidence, data.shielded_confidence);
            showStage(stage3);
        }
        btnGenerateShield.innerText = "GENERATE ADVERSARIAL PERTURBATION";
        btnGenerateShield.disabled = false;
    });

    function renderChart(baseline, shielded) {
        const ctx = document.getElementById("confidenceChart").getContext("2d");
        if (confidenceChart) confidenceChart.destroy();

        confidenceChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: ["AI Model confidence: 99%", "9%"],
                datasets: [{
                    data: [baseline, shielded],
                    borderColor: "#6366f1",
                    backgroundColor: "rgba(99, 102, 241, 0.2)",
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointBackgroundColor: ["#6366f1", "#6366f1"],
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: { color: "#64748b", font: { size: 9 } },
                        grid: { color: "rgba(51, 65, 85, 0.3)" }
                    },
                    x: {
                        ticks: { color: "#94a3b8", font: { size: 9 } },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    btnExportPurge.addEventListener("click", () => {
        window.location.href = `/api/export?purge=${purgeCheckbox.checked}`;
    });

    btnReset.addEventListener("click", async () => {
        await fetch("/api/purge", { method: "POST" });
        currentFile = null;
        fileInput.value = "";
        uploadLabel.innerText = "UPLOAD RAW MEDIA";
        btnStartScan.disabled = true;
        showStage(stage1);
    });
});
