const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const resultContainer = document.getElementById('result');

fileInput.addEventListener('change', () => {
  previewContainer.innerHTML = '';
  const file = fileInput.files[0];
  if (file) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    previewContainer.appendChild(img);
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    alert('Please select an image first.');
    return;
  }

  const fd = new FormData();
  fd.append('image', file);
  resultContainer.innerHTML = 'Analyzing...';

  try {
    const res = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: fd
    });
    const data = await res.json();
    resultContainer.innerHTML = `
      <h3>Result</h3>
      <p><strong>Diagnosis:</strong> ${data.class_name} (${(data.probability*100).toFixed(1)}%)</p>
      <h4>Prescriptions</h4>
      <ul>${data.prescriptions.map(p => `<li>${p}</li>`).join('')}</ul>
    `;
  } catch (err) {
    console.error(err);
    resultContainer.innerHTML = '<p style="color:red;">Prediction failed. Check backend.</p>';
  }
});
