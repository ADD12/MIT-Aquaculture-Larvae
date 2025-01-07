const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
app.use(cors());

// Set up Multer for handling file uploads
const upload = multer({ dest: 'uploads/' });

app.post('/process-image', upload.single('image'), (req, res) => {
  if (!req.file) {
    console.error('No file uploaded');
    return res.status(400).send('No file uploaded.');
  }
  console.log(`Received file: ${req.file.path}`);

  const imagePath = path.resolve(req.file.path);
  const scriptPath = path.join(__dirname, 'process_image.py');  // Corrected path

  // Log paths for debugging
  console.log(`Script path: ${scriptPath}`);
  console.log(`Image path: ${imagePath}`);

  // Call Python script to process the image
  const python = spawn('python3', [scriptPath, imagePath]);

  let dataString = '';

  // Capture stdout from Python script
  python.stdout.on('data', (data) => {
    dataString += data.toString();
  });

  // Capture stderr from Python script
  python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  // Handle close event and parse the output
  python.on('close', (code) => {
    // Log raw output for debugging
    //console.log('Raw Python output:', dataString);

    if (code !== 0) {
      console.error('Error processing image.');
      return res.status(500).send('Error processing image.');
    }

    // Split the output by lines and try to parse only the last line
    const lines = dataString.trim().split('\n');
    const lastLine = lines[lines.length - 1];

    // Attempt to parse the last line as JSON
    try {
      const result = JSON.parse(lastLine);
      console.log('Processed result:', { ...result, annotatedImageBase64: '[base64 data]' });  // Hide base64 data for logging
      res.json(result);  // Send the parsed result to the frontend
    } catch (error) {
      console.error('Error parsing Python output:', error);
      console.error('Received data:', dataString);  // Log what was received
      return res.status(500).send('Invalid output from the Python script.');
    }
  });
});

const PORT = process.env.PORT || 3001;  // Updated to port 3001
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
