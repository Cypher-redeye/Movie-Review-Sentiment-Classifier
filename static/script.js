document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("analyzeBtn");
    const btnText = btn.querySelector(".btn-text");
    const textarea = document.getElementById("reviewInput");
    const resultContainer = document.getElementById("resultContainer");
    const resultBadge = document.getElementById("resultBadge");
    const confidenceNumber = document.getElementById("confidenceNumber");
    const resultDescription = document.getElementById("resultDescription");

    textarea.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            btn.click();
        }
    });

    // Helper to animate numbers gracefully
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const currentVal = start + easeOut * (end - start);
            obj.innerHTML = currentVal.toFixed(1);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    btn.addEventListener("click", async () => {
        const text = textarea.value.trim();
        if (!text) {
            textarea.focus();
            return;
        }

        btn.disabled = true;
        btnText.textContent = "Analyzing...";
        
        // Hide result to re-trigger reveal
        resultContainer.classList.add("hide");

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (!response.ok) {
                alert(data.error || "An error occurred.");
                return;
            }

            resultBadge.textContent = data.label;
            
            resultContainer.classList.remove("theme-pos", "theme-neg");
            resultContainer.classList.add(data.label === "Positive" ? "theme-pos" : "theme-neg");

            if (data.label === "Positive") {
                resultDescription.textContent = "The neural network has classified this text as having a predominantly favorable emotional polarity.";
            } else {
                resultDescription.textContent = "The neural network has classified this text as having a predominantly critical or unfavorable emotional polarity.";
            }

            // Reveal the result section gracefully
            setTimeout(() => {
                resultContainer.classList.remove("hide");
                animateValue(confidenceNumber, 0, data.confidence * 100, 1500);
            }, 300); // Slight delay for elegance

        } catch (error) {
            console.error("Fetch Error:", error);
            alert("Failed to connect to the server.");
        } finally {
            btn.disabled = false;
            btnText.textContent = "Analyze Text";
        }
    });
});
