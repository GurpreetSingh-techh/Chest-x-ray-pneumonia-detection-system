document.addEventListener("DOMContentLoaded", function () {

    // =====================================
    // GET HTML ELEMENTS
    // =====================================

    const imageInput = document.getElementById("imageInput");
    const dropArea = document.getElementById("dropArea");
    const browseButton = document.getElementById("browseButton");
    const uploadContent = document.getElementById("uploadContent");
    const previewContainer = document.getElementById("previewContainer");
    const previewImage = document.getElementById("previewImage");
    const fileName = document.getElementById("fileName");
    const removeImage = document.getElementById("removeImage");

    const analyzeButton = document.getElementById("analyzeButton");
    const loadingText = document.getElementById("loadingText");
    const errorMessage = document.getElementById("errorMessage");

    const emptyResult = document.getElementById("emptyResult");
    const resultContent = document.getElementById("resultContent");

    const predictionText = document.getElementById("predictionText");
    const resultTitle = document.getElementById("resultTitle");
    const resultDescription = document.getElementById("resultDescription");

    const normalScore = document.getElementById("normalScore");
    const pneumoniaScore = document.getElementById("pneumoniaScore");

    const normalBar = document.getElementById("normalBar");
    const pneumoniaBar = document.getElementById("pneumoniaBar");

    const gradcamSection = document.getElementById("gradcamSection");
    const gradcamImage = document.getElementById("gradcamImage");

    let selectedFile = null;
    let previewURL = null;


    // =====================================
    // ERROR MESSAGE
    // =====================================

    function showError(message) {

        errorMessage.textContent = message;
        errorMessage.classList.remove("hidden");

        console.error("ClarityX:", message);
    }


    function clearError() {

        errorMessage.textContent = "";
        errorMessage.classList.add("hidden");

    }


    // =====================================
    // RESET RESULTS
    // =====================================

    function resetResults() {

        resultContent.classList.add("hidden");
        emptyResult.classList.remove("hidden");

        gradcamSection.classList.add("hidden");

        gradcamImage.removeAttribute("src");

        normalBar.style.width = "0%";
        pneumoniaBar.style.width = "0%";

    }


    // =====================================
    // IMAGE VALIDATION
    // =====================================

    function validateFile(file) {

        if (!file) return false;

        const allowedTypes = [
            "image/jpeg",
            "image/png"
        ];

        if (!allowedTypes.includes(file.type)) {

            showError("Please upload a JPG or PNG image.");

            return false;
        }

        if (file.size > 10 * 1024 * 1024) {

            showError("Maximum image size is 10 MB.");

            return false;
        }

        return true;
    }


    // =====================================
    // DISPLAY IMAGE
    // =====================================

    function displayImage(file) {

        if (!validateFile(file)) return;

        selectedFile = file;

        imageInput.value = "";

        clearError();
        resetResults();

        if (previewURL) {
            URL.revokeObjectURL(previewURL);
        }

        previewURL = URL.createObjectURL(file);

        previewImage.src = previewURL;

        previewContainer.classList.remove("hidden");

        uploadContent.classList.add("hidden");

        fileName.textContent = file.name;

        analyzeButton.disabled = false;

    }


    // =====================================
    // BROWSE IMAGE
    // =====================================

    browseButton.addEventListener("click", function (event) {

        event.preventDefault();

        imageInput.click();

    });


    // =====================================
    // IMAGE SELECTION
    // =====================================

    imageInput.addEventListener("change", function () {

        const file = imageInput.files[0];

        if (file) {

            displayImage(file);

        }

    });


    // =====================================
    // DRAG AND DROP
    // =====================================

    dropArea.addEventListener("dragover", function (event) {

        event.preventDefault();

        dropArea.classList.add("drag-over");

    });


    dropArea.addEventListener("dragleave", function () {

        dropArea.classList.remove("drag-over");

    });


    dropArea.addEventListener("drop", function (event) {

        event.preventDefault();

        dropArea.classList.remove("drag-over");

        const file = event.dataTransfer.files[0];

        if (file) {

            displayImage(file);

        }

    });


    // =====================================
    // CLICK UPLOAD AREA
    // =====================================

    dropArea.addEventListener("click", function (event) {

        if (
            event.target.closest("button") ||
            event.target.closest("input")
        ) {
            return;
        }

        if (!selectedFile) {

            imageInput.click();

        }

    });


    // =====================================
    // REMOVE IMAGE
    // =====================================

    removeImage.addEventListener("click", function (event) {

        event.preventDefault();

        selectedFile = null;

        imageInput.value = "";

        previewImage.removeAttribute("src");

        previewContainer.classList.add("hidden");

        uploadContent.classList.remove("hidden");

        fileName.textContent = "";

        analyzeButton.disabled = true;

        clearError();

        resetResults();

    });


    // =====================================
    // LOADING
    // =====================================

    function showLoading() {

        loadingText.classList.remove("hidden");

        loadingText.textContent =
            "Analyzing your X-ray. Please wait...";

        analyzeButton.disabled = true;

        analyzeButton.textContent = "Analyzing...";

    }


    function hideLoading() {

        loadingText.classList.add("hidden");

        analyzeButton.disabled = !selectedFile;

        analyzeButton.innerHTML =
            'Analyze X-ray <span>→</span>';

    }


    // =====================================
    // DISPLAY RESULTS
    // =====================================

    function displayResults(data) {

        predictionText.textContent = data.prediction;

        resultTitle.textContent =
            data.prediction === "PNEUMONIA"
                ? "Pneumonia detected by the model"
                : "Normal classification";

        resultDescription.textContent =
            data.description || "";


        // Scores

        const normal = Number(data.normal_score);
        const pneumonia = Number(data.pneumonia_score);

        if (
            !Number.isFinite(normal) ||
            !Number.isFinite(pneumonia)
        ) {
            throw new Error("Invalid prediction scores received.");
        }

        normalScore.textContent = normal + "%";

        pneumoniaScore.textContent = pneumonia + "%";


        // Progress bars

        normalBar.style.width = normal + "%";

        pneumoniaBar.style.width = pneumonia + "%";


        // Show results by removing hidden class

        emptyResult.classList.add("hidden");

        resultContent.classList.remove("hidden");


        // Grad-CAM

        if (data.gradcam) {

            let imageData = data.gradcam;

            if (!imageData.startsWith("data:image")) {

                imageData =
                    "data:image/png;base64," + imageData;

            }

            gradcamImage.src = imageData;

            gradcamSection.classList.remove("hidden");

        } else {

            gradcamSection.classList.add("hidden");

        }

    }


    // =====================================
    // ANALYZE X-RAY
    // =====================================

    analyzeButton.addEventListener("click", async function (event) {

        event.preventDefault();

        clearError();

        if (!selectedFile) {

            showError("Please upload an X-ray image first.");

            return;
        }

        showLoading();

        resetResults();

        const formData = new FormData();

        formData.append("image", selectedFile);


        try {

            console.log("Sending image to Flask...");

            const response = await fetch("/predict", {

                method: "POST",

                body: formData,

                signal: AbortSignal.timeout(240000)

            });


            const data = await response.json();

            console.log("Flask response:", data);


            if (!response.ok) {

                throw new Error(
                    data.error || "Prediction failed."
                );

            }


            if (!data.prediction) {

                throw new Error(
                    "No prediction was received."
                );

            }


            displayResults(data);

            console.log(
                "ClarityX prediction:",
                data.prediction
            );


        } catch (error) {

            console.error("Prediction error:", error);

            showError(
                error.message || "Unable to analyze the image."
            );

        } finally {

            hideLoading();

        }

    });


    // =====================================
    // INITIAL STATE
    // =====================================

    analyzeButton.disabled = true;

    loadingText.classList.add("hidden");

    clearError();

    console.log("ClarityX JavaScript loaded successfully.");

});
