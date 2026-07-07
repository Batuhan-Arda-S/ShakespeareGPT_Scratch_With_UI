const form = document.querySelector("#generateForm");
const promptInput = document.querySelector("#prompt");
const output = document.querySelector("#output");
const button = document.querySelector("#generateButton");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  button.disabled = true;
  button.textContent = "Generating...";
  output.textContent = "Generating...";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: promptInput.value,
        max_tokens: 750,
        temperature: 0.8,
      }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Generation failed.");
    }

    output.textContent = data.text;
  } catch (error) {
    output.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Generate";
  }
});
