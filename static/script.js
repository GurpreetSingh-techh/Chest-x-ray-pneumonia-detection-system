document.addEventListener("DOMContentLoaded", function () {

    // ==========================================
    // 1. GET HTML ELEMENTS
    // ==========================================

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

    const gradcamSection = document.getElementById("gradcamSection");

    const gradcamImage = document.getElementById("gradcamImage");


    // ==========================================
    // 2. CHECK ELEMENTS
    // ==========================================

    if (!imageInput) {
        console.error("ClarityX: imageInput not found.");
        return;
    }

    let selectedFile = null;


    // ==========================================
    // 3. FILE VALIDATION
    // ==========================================

    function validateFile(file) {

        if (!file) {
            return false;
        }

        const allowedTypes = [
            "image/jpeg",
            "image/png"
        ];

        if (!allowedTypes.includes(file.type)) {

            showError("Please upload a JPG or PNG image.");

            return false;
        }

        if (file.size > 10 * 1024 * 1024) {

            showError("Image size must be smaller than 10 MB.");

            return false;
        }

        return true;
    }


    // ==========================================
    // 4. SHOW ERROR
    // ==========================================

    function showError(message) {

        if (errorMessage) {

            errorMessage.textContent = message;

            errorMessage.style.display = "block";
        }

        console.error("ClarityX:", message);
    }


    // ==========================================
    // 5. CLEAR ERROR
    // ==========================================

    function clearError() {

        if (errorMessage) {

            errorMessage.textContent = "";

            errorMessage.style.display = "none";
        }
    }


    // ==========================================
    // 6. RESET RESULTS
    // ==========================================

    function resetResults() {

        if (resultContent) {
            resultContent.style.display = "none";
        }

        if (emptyResult) {
            emptyResult.style.display = "block";
        }

        if (gradcamSection) {
            gradcamSection.style.display = "none";
        }

        if (gradcamImage) {
            gradcamImage.removeAttribute("src");
        }
    }


    // ==========================================
    // 7. DISPLAY IMAGE PREVIEW
    // ==========================================

    function displayImage(file) {

        if (!validateFile(file)) {
            return;
        }

        selectedFile = file;

        clearError();

        resetResults();

        const reader = new FileReader();

        reader.onload = function (event) {

            if (previewImage) {

                previewImage.src = event.target.result;

                previewImage.style.display = "block";
            }

            if (previewContainer) {

                previewContainer.style.display = "block";
            }

            if (uploadContent) {

                uploadContent.style.display = "none";
            }

            if (fileName) {

                fileName.textContent = file.name;
            }

            if (analyzeButton) {

                analyzeButton.disabled = false;
            }

        };

        reader.onerror = function () {

            showError("Unable to read this image. Please try another.");

        };

        reader.readAsDataURL(file);

    }


    // ==========================================
    // 8. BROWSE IMAGE BUTTON
    // ==========================================

    if (browseButton) {

        browseButton.addEventListener("click", function (event) {

            // Prevent accidental form submission.

            event.preventDefault();

            // If the browse button is a label connected
            // to the input, let the browser handle it.

            if (
                browseButton.tagName.toLowerCase() === "label" &&
                browseButton.htmlFor === "imageInput"
            ) {
                return;
            }

            imageInput.click();

        });

    }


    // ==========================================
    // 9. FILE INPUT CHANGE
    // ==========================================

    imageInput.addEventListener("change", function () {

        const file = imageInput.files[0];

        if (file) {

            displayImage(file);

        }

    });


    // ==========================================
    // 10. DRAG AND DROP
    // ==========================================

    if (dropArea) {

        dropArea.addEventListener("dragover", function (event) {

            event.preventDefault();

            dropArea.classList.add("drag-over");

        });


        dropArea.addEventListener("dragleave", function (event) {

            event.preventDefault();

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


        // Allow clicking the upload area.

        dropArea.addEventListener("click", function (event) {

            if (
                event.target.closest("button") ||
                event.target.closest("label") ||
                event.target.closest("input")
            ) {
                return;
            }

            if (!selectedFile) {

                imageInput.click();

            }

        });

    }


    // ==========================================
    // 11. REMOVE IMAGE
    // ==========================================

    if (removeImage) {

        removeImage.addEventListener("click", function (event) {

            event.preventDefault();

            selectedFile = null;

            imageInput.value = "";

            if (previewImage) {

                previewImage.removeAttribute("src");

                previewImage.style.display = "none";
            }

            if (previewContainer) {

                previewContainer.style.display = "none";
            }

            if (uploadContent) {

                uploadContent.style.display = "block";
            }

            if (fileName) {

                fileName.textContent = "";
            }

            if (analyzeButton) {

                analyzeButton.disabled = true;
            }

            clearError();

            resetResults();

        });

    }


    // ==========================================
    // 12. SHOW LOADING
    // ==========================================

    function showLoading() {

        if (loadingText) {

            loadingText.style.display = "block";

            loadingText.textContent =
                "Analyzing your X-ray with AI. Please wait...";

        }

        if (analyzeButton) {

            analyzeButton.disabled = true;

            analyzeButton.textContent = "Analyzing...";

        }

    }


    // ==========================================
    // 13. HIDE LOADING
    // ==========================================

    function hideLoading() {

        if (loadingText) {

            loadingText.style.display = "none";

        }

        if (analyzeButton) {

            analyzeButton.disabled = false;

            analyzeButton.textContent = "Analyze X-ray";

        }

    }


    // ==========================================
    // 14. DISPLAY PREDICTION
    // ==========================================

    function displayResults(data) {

        // Prediction

        const predictionElement =
            document.getElementById("predictionText") ||
            document.getElementById("predictionResult") ||
            document.getElementById("resultPrediction");

        if (predictionElement) {

            predictionElement.textContent = data.prediction;

        }


        // Normal score

        const normalElement =
            document.getElementById("normalScore");

        if (normalElement) {

            normalElement.textContent =
                data.normal_score + "%";

        }


        // Pneumonia score

        const pneumoniaElement =
            document.getElementById("pneumoniaScore");

        if (pneumoniaElement) {

            pneumoniaElement.textContent =
                data.pneumonia_score + "%";

        }


        // Description

        const descriptionElement =
            document.getElementById("description");

        if (descriptionElement) {

            descriptionElement.textContent =
                data.description || "";

        }


        // Show results

        if (emptyResult) {

            emptyResult.style.display = "none";

        }

        if (resultContent) {

            resultContent.style.display = "block";

        }


        // ======================================
        // GRAD-CAM DISPLAY
        // ======================================

        if (data.gradcam && gradcamImage && gradcamSection) {

            let imageData = data.gradcam;

            if (!imageData.startsWith("data:image")) {

                imageData = "data:image/png;base64," + imageData;

            }

            gradcamImage.src = imageData;

            gradcamSection.style.display = "block";

        } else {

            if (gradcamSection) {

                gradcamSection.style.display = "none";

            }

        }

    }


    // ==========================================
    // 15. ANALYZE IMAGE
    // ==========================================

    if (analyzeButton) {

        analyzeButton.addEventListener("click", async function (event) {

            event.preventDefault();

            clearError();

            if (!selectedFile) {

                showError("Please upload an X-ray image first.");

                return;

            }


            // Show loading

            showLoading();

            resetResults();


            // Create FormData

            const formData = new FormData();

            formData.append("image", selectedFile);


            try {

                console.log("Sending image to Flask...");


                // Send image to Flask backend

                const response = await fetch("/predict", {

                    method: "POST",

                    body: formData,

                    signal: AbortSignal.timeout(180000)

                });


                console.log("Server response:", response.status);


                // Read response

                const responseText = await response.text();

                if (!responseText) {

                    throw new Error(
                        "The server returned an empty response."
                    );

                }

                let data;

                try {

                    data = JSON.parse(responseText);

                } catch (parseError) {

                    console.error("Invalid server response:", responseText);

                    throw new Error(
                        "The server returned an invalid response."
                    );

                }


                // Check backend errors

                if (!response.ok) {

                    throw new Error(
                        data.error || "Prediction failed."
                    );

                }


                if (!data.prediction) {

                    throw new Error(
                        "Prediction result was not received."
                    );

                }


                // Display prediction

                displayResults(data);

                console.log("Prediction:", data.prediction);


            }

            catch (error) {

                console.error("ClarityX prediction error:", error);

                showError(
                    error.message || "Unable to analyze the image."
                );

            }

            finally {

                hideLoading();

            }

        });

    }


    // ==========================================
    // 16. INITIAL WEBSITE STATE
    // ==========================================

    if (analyzeButton) {

        analyzeButton.disabled = !selectedFile;

    }

    if (loadingText) {

        loadingText.style.display = "none";

    }

    clearError();

    console.log("ClarityX JavaScript loaded successfully.");

});
