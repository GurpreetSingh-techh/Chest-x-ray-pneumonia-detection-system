const imageInput = document.getElementById("imageInput");

const browseButton = document.getElementById("browseButton");

const uploadArea = document.getElementById("uploadArea");

const previewContainer = document.getElementById("previewContainer");

const previewImage = document.getElementById("previewImage");

const removeButton = document.getElementById("removeButton");

const analyzeButton = document.getElementById("analyzeButton");

const loading = document.getElementById("loading");

const resultBox = document.getElementById("resultBox");

const errorBox = document.getElementById("errorBox");

let selectedFile = null;


// ----------------------------------
// Browse Image
// ----------------------------------

browseButton.addEventListener("click", () => {

    imageInput.click();

});


// ----------------------------------
// Select Image
// ----------------------------------

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (file) {

        handleFile(file);

    }

});


// ----------------------------------
// Handle Selected Image
// ----------------------------------

function handleFile(file) {

    hideMessages();

    if (!file.type.startsWith("image/")) {

        showError("Please select a valid image.");

        return;

    }

    if (file.size > 10 * 1024 * 1024) {

        showError("Image size must be less than 10 MB.");

        return;

    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src = event.target.result;

        previewContainer.style.display = "block";

        analyzeButton.disabled = false;

        uploadArea.style.display = "none";

    };

    reader.readAsDataURL(file);

}


// ----------------------------------
// Drag and Drop
// ----------------------------------

uploadArea.addEventListener("dragover", function (event) {

    event.preventDefault();

    uploadArea.classList.add("dragover");

});


uploadArea.addEventListener("dragleave", function () {

    uploadArea.classList.remove("dragover");

});


uploadArea.addEventListener("drop", function (event) {

    event.preventDefault();

    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (file) {

        handleFile(file);

    }

});


// ----------------------------------
// Remove Image
// ----------------------------------

removeButton.addEventListener("click", resetPage);


// ----------------------------------
// Analyze Image
// ----------------------------------

analyzeButton.addEventListener("click", async function () {

    if (!selectedFile) {

        showError("Please select an X-ray image.");

        return;

    }

    hideMessages();

    analyzeButton.style.display = "none";

    loading.style.display = "block";

    const formData = new FormData();

    formData.append("image", selectedFile);


    try {

        const response = await fetch("/predict", {

            method: "POST",

            body: formData

        });

        const data = await response.json();

        loading.style.display = "none";

        if (!response.ok) {

            showError(data.error || "Prediction failed.");

            analyzeButton.style.display = "block";

            return;

        }

        showResult(data);

    }

    catch (error) {

        loading.style.display = "none";

        analyzeButton.style.display = "block";

        showError(
            "Something went wrong. Please try again."
        );

        console.error(error);

    }

});


// ----------------------------------
// Show Prediction Result
// ----------------------------------

function showResult(data) {

    resultBox.style.display = "block";

    const predictionText =
        document.getElementById("predictionText");

    const description =
        document.getElementById("description");

    const normalScore =
        document.getElementById("normalScore");

    const pneumoniaScore =
        document.getElementById("pneumoniaScore");

    const normalBar =
        document.getElementById("normalBar");

    const pneumoniaBar =
        document.getElementById("pneumoniaBar");


    predictionText.textContent = data.prediction;

    predictionText.classList.remove("pneumonia");

    if (data.prediction === "PNEUMONIA") {

        predictionText.classList.add("pneumonia");

    }

    description.textContent = data.description;

    normalScore.textContent =
        data.normal_score + "%";

    pneumoniaScore.textContent =
        data.pneumonia_score + "%";


    normalBar.style.width =
        data.normal_score + "%";

    pneumoniaBar.style.width =
        data.pneumonia_score + "%";


    resultBox.scrollIntoView({

        behavior: "smooth",

        block: "center"

    });

}


// ----------------------------------
// Show Error
// ----------------------------------

function showError(message) {

    errorBox.textContent = message;

    errorBox.style.display = "block";

}


// ----------------------------------
// Hide Messages
// ----------------------------------

function hideMessages() {

    errorBox.style.display = "none";

    resultBox.style.display = "none";

    loading.style.display = "none";

}


// ----------------------------------
// Reset Page
// ----------------------------------

function resetPage() {

    selectedFile = null;

    imageInput.value = "";

    previewImage.src = "";

    previewContainer.style.display = "none";

    uploadArea.style.display = "block";

    analyzeButton.disabled = true;

    analyzeButton.style.display = "block";

    hideMessages();

    document.getElementById("analyzer").scrollIntoView({

        behavior: "smooth"

    });

}
