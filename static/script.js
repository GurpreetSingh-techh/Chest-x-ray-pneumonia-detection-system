document.addEventListener("DOMContentLoaded", function () {

    // =================================
    // GET HTML ELEMENTS
    // =================================

    const imageInput = document.getElementById("imageInput");

    const previewImage = document.getElementById("previewImage");

    const analyzeBtn = document.getElementById("analyzeBtn");

    const statusMessage = document.getElementById("statusMessage");

    const resultSection = document.getElementById("resultSection");

    const predictionText = document.getElementById("predictionText");

    const normalScore = document.getElementById("normalScore");

    const pneumoniaScore = document.getElementById("pneumoniaScore");

    const description = document.getElementById("description");


    // =================================
    // CHECK HTML ELEMENTS
    // =================================

    if (
        !imageInput ||
        !analyzeBtn ||
        !statusMessage ||
        !resultSection
    ) {

        console.error(
            "ClarityX: Required HTML elements are missing."
        );

        return;
    }


    // =================================
    // IMAGE PREVIEW
    // =================================

    imageInput.addEventListener("change", function () {

        const file = imageInput.files[0];

        if (!file) {
            return;
        }

        if (!["image/jpeg", "image/png"].includes(file.type)) {

            statusMessage.textContent =
                "Please upload a JPG or PNG image.";

            imageInput.value = "";

            return;
        }

        if (file.size > 10 * 1024 * 1024) {

            statusMessage.textContent =
                "Image must be smaller than 10 MB.";

            imageInput.value = "";

            return;
        }

        const imageURL = URL.createObjectURL(file);

        if (previewImage) {

            previewImage.src = imageURL;

            previewImage.style.display = "block";

        }

        resultSection.style.display = "none";

        statusMessage.textContent =
            "Image uploaded successfully. Ready for analysis.";

    });


    // =================================
    // ANALYZE IMAGE
    // =================================

    analyzeBtn.addEventListener("click", async function () {

        const file = imageInput.files[0];

        if (!file) {

            statusMessage.textContent =
                "Please upload an X-ray image first.";

            return;

        }

        // Disable button

        analyzeBtn.disabled = true;

        analyzeBtn.textContent = "Analyzing...";

        statusMessage.textContent =
            "Connecting to AI model. Please wait...";

        resultSection.style.display = "none";


        // Create form data

        const formData = new FormData();

        formData.append("image", file);


        try {

            // Send image to Flask

            statusMessage.textContent =
                "Processing your X-ray image...";

            const response = await fetch("/predict", {

                method: "POST",

                body: formData,

                signal: AbortSignal.timeout(180000)

            });


            // Read response

            const data = await response.json();


            // Check server errors

            if (!response.ok) {

                throw new Error(
                    data.error || "Prediction failed."
                );

            }


            // =================================
            // SHOW PREDICTION
            // =================================

            if (predictionText) {

                predictionText.textContent =
                    data.prediction;

            }


            // =================================
            // SHOW NORMAL SCORE
            // =================================

            if (normalScore) {

                normalScore.textContent =
                    data.normal_score + "%";

            }


            // =================================
            // SHOW PNEUMONIA SCORE
            // =================================

            if (pneumoniaScore) {

                pneumoniaScore.textContent =
                    data.pneumonia_score + "%";

            }


            // =================================
            // SHOW DESCRIPTION
            // =================================

            if (description) {

                description.textContent =
                    data.description;

            }


            // =================================
            // DISPLAY RESULT
            // =================================

            resultSection.style.display = "block";

            statusMessage.textContent =
                "Analysis completed successfully!";

        }

        catch (error) {

            console.error(
                "ClarityX Error:",
                error
            );

            if (
                error.name === "TimeoutError" ||
                error.name === "AbortError"
            ) {

                statusMessage.textContent =
                    "Request timed out. The server may be loading the AI model. Please try again.";

            } else {

                statusMessage.textContent =
                    error.message || "Something went wrong.";

            }

        }

        finally {

            analyzeBtn.disabled = false;

            analyzeBtn.textContent = "Analyze X-ray";

        }

    });

});
