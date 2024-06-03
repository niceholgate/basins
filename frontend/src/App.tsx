import React, { useState } from "react";
import ImageRenderer from './components/ImageRenderer';
import { Player } from 'video-react';
import ReactPlayer from 'react-player'
import './App.css';

function App() {
    const [exp1, setExp1] = useState('y**2+2*x*y+2*x**2-4-0.5*sin(15*x)');
    const [exp2, setExp2] = useState('y-10*x**2+3');
    const [colourScheme, setColourScheme] = useState(1);
    const [uuid, setUuid] = useState('');
    const [rgbArray, setRgbArray] = useState([[]]);
    const [message, setMessage] = useState('');

    const hostname = window.location.hostname === 'localhost' ? 'localhost' : '192.168.0.106';

    const generateArray = async () => {
        await fetch(`http://${hostname}:8000/create/still`, {
            method: 'POST',
            headers: {
              'Accept': 'application/json, text/plain, */*',
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({color_set: colourScheme, x_pixels: 500, y_pixels: 300, expressions: [exp1, exp2]})
          })
            .then((res) => {
                if (res.status!==202) {
                    throw new Error(`Could not generate image`)
                }
                return res.json()
            })
            .then(data => {
                setMessage(`Generating an image with UUID=${data.id}`)
            })
            .catch((error) => {
                setMessage('Could not generate requested image')
            })
    };

    const loadArray = async () => {
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
                setMessage('Could not retrieve requested image data')
                setRgbArray([[]])
            })
    };

    const handleGenerateClick = async () => {
        generateArray();
    }

    const handleLoadClick = async () => {
        loadArray();
    }

    function checkArray(rgbArray: number[][]) {
        if (rgbArray.length > 0) {
            if (rgbArray[0].length > 0) {
                return <ImageRenderer rgbArray={rgbArray} />
            }
        }
        return null;
    }


    const range = (start: number, end: number) => Array.from({length: (end - start)}, (v, k) => k + start);

    return (
      <div>
            <h1>Basins of Attraction</h1>
            
            <h2>Create Image</h2>
            <div className="flex-container">
                <div>
                    <label>Expressions</label>
                    <div><input type='text' value={exp1} onChange={e => setExp1(e.target.value)} title={'Expression 1'}/></div>
                    <div><input type='text' value={exp2} onChange={e => setExp2(e.target.value)} title={'Expression 2'}/></div>
                </div>
                <div>
                    {'todo: change this to '}
                    <label>Colour scheme</label>
                    <div><select name="colourScheme" id="colourScheme" value={colourScheme} onChange={e => setColourScheme(Number(e.target.value))}>
                        {range(1, 11).map((x) => (
                            <option value={x}>{x}</option>
                        ))}
                    </select></div>    
                </div>
                <div>
                    <button type='submit' onClick={handleGenerateClick}>Generate</button>
                </div>
            </div>

            <h2>Load Image</h2>
            <input type='text' value={uuid} onChange={e => setUuid(e.target.value)} placeholder='Enter an ID'/>
            <div>
                <button type='submit' onClick={handleLoadClick}>Load</button>
            </div>

          
          <div>
            <label>Message={message}</label>
            <center>{checkArray(rgbArray)}</center>
            
            <ReactPlayer url={`http://${hostname}:8000/video`}/>
            <Player src={`http://${hostname}:8000/video`}/>
          </div>
      </div>
    );
};

export default App;
