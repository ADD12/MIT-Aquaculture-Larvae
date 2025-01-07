import React, { useState } from 'react';
import axios from 'axios';
import './Home.css';

const Home = () => {
  const [images, setImages] = useState([]);
  const [imagePreviews, setImagePreviews] = useState([]);
  const [results, setResults] = useState([]);
  const [averageAlive, setAverageAlive] = useState(0);
  const [expectedLarvaeCount, setExpectedLarvaeCount] = useState(0);  // New state
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState('');

  const handleImageChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 30) {
      alert('You can upload a maximum of 30 images.');
      return;
    }
    setImages(files);

    // Set image previews
    const previews = files.map((file) => URL.createObjectURL(file));
    setImagePreviews(previews);
  };

  const handleSubmit = async () => {
    if (images.length === 0) {
      alert('Please upload images to process.');
      return;
    }

    setLoading(true);
    let totalAlive = 0;
    const resultsArray = [];

    // Loop through each image and send individual requests
    for (let i = 0; i < images.length; i++) {
      const image = images[i];
      const formData = new FormData();
      formData.append('image', image);

      try {
        // Update progress state
        setProgress(`Processing image ${i + 1} of ${images.length}...`);

        const response = await axios.post('http://localhost:3001/process-image', formData, {
          timeout: 10000,  // Set timeout to 10 seconds
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        resultsArray.push(response.data);
        totalAlive += response.data.objectCount;  // Use objectCount for total alive count
      } catch (error) {
        console.error('Error processing image:', error);
        if (error.code === 'ECONNABORTED') {
          alert('Request timed out. Please try again.');
        } else {
          alert('Error processing image. Please try again.');
        }
      }
    }

    // Calculate average alive larvae count
    const avgAlive = totalAlive / images.length;
    setAverageAlive(avgAlive);

    // Calculate expected larvae count with multiplier
    setExpectedLarvaeCount(avgAlive * 28);

    setResults(resultsArray);
    setLoading(false);
    setProgress('All images processed.');
  };

  return (
    <div className="home">
      <div className="title">
        <h1>Shellfish Specimen Counter</h1>
        <p>Upload up to 30 images from your microscope and then click "Run Model".</p>
      </div>
      <div className="home-container">
        <div className="left-section">
          <div className="upload-box">
            <h2>Upload Images</h2>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleImageChange}
            />
          </div>

          <div className="image-grid">
            {imagePreviews.map((src, index) => (
              <div key={index} className="image-item">
                <p>Image {index + 1}</p>
                <img src={src} alt={`uploaded-${index}`} />
              </div>
            ))}
          </div>
        </div>

        <div className="right-section">
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="run-button"
          >
            {loading ? 'Running Model...' : 'Run Model'}
          </button>

          {loading && <p className="progress-text">{progress}</p>}

          {results.length > 0 && (
            <div className="results-section">
              <h3>Results Analysis</h3>
              <p>Average Alive Larvae: {averageAlive.toFixed(2)}</p>
              <p>Expected Total Larvae Count: {expectedLarvaeCount.toFixed(2)}</p>  {/* New display */}

              <div className="annotated-image-grid">
                {results.map((result, index) => (
                  <div key={index} className="image-item">
                    <p>Image {index + 1}</p>
                    <p>Larvae count: {result.objectCount}</p>
                    <img
                      src={`data:image/jpeg;base64,${result.annotatedImageBase64}`}
                      alt={`annotated-${index}`}
                      className="annotated-image"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;
