const resetButton = document.getElementById('resetButton');
resetButton.addEventListener('click', function() {
  uploadInput.value = '';
  previewElement.innerHTML = '';
});
