const uploadInput = document.getElementById('imageInput');
const previewElement = document.getElementById('preview');

uploadInput.addEventListener('change', handleFileUpload);

function handleFileUpload(event) {
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = document.createElement('img');
    img.src = e.target.result;

    previewElement.innerHTML = '';
    previewElement.appendChild(img);
  };
  reader.readAsDataURL(file);
}

