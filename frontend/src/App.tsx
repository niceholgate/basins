import React, { useState } from "react";
import ImageRenderer from './components/ImageRenderer';
import './App.css';

function App() {
    const [uuid, setUuid] = useState('')
    const [rgbArray, setRgbArray] = useState([[]]);

    const hostname = window.location.hostname === 'localhost' ? 'localhost' : '192.168.0.106'

    const fetchArray = async () => {
        await fetch(`http://${hostname}:8000/load/rgb_frame/${uuid}/0`)
            .then((res) => {
                if (!res.ok) {
                    throw new Error(`Could not retrieve image with uuid=${uuid}`)
                }
                return res.json()
            })
            .then(data => {
                setRgbArray(data)
            })
            .catch((error) => {
                setUuid('Could not retrieve requested image data')
                setRgbArray([[]])
            })
    };

    const handleClick = async () => {
        fetchArray();
    }

    function checkArray(rgbArray: number[][]) {
        if (rgbArray.length > 0) {
            if (rgbArray[0].length > 0) {
                return <ImageRenderer rgbArray={rgbArray} />
            }
        }
        return null;
    }

    return (
      <div>
          <h1>Basins of Attraction</h1>
          <center>
            <input type='text' value={uuid} onChange={e => setUuid(e.target.value)} placeholder='Enter an ID'/>
            <button type='submit' onClick={handleClick}>Search</button>
          </center>
          
          <div>
            <center>{checkArray(rgbArray)}</center>
          </div>
      </div>
    );
};

export default App;
