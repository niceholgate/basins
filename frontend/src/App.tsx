import React, { useState, useEffect } from "react";
import ImageRenderer from './components/ImageRenderer';
import SliderStepInput from './components/SliderStepInput';
import './App.css';
import {
    ChakraProvider,
    Box,
    VStack,
    Spacer,
    Divider,
    Flex,
    Button,
    Center,
    Input,
    Select } from "@chakra-ui/react";

type Cache = {
    [key: number]: number[][]
}

function App() {
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

    const loadArray = async (newFrame: number) => {
        return await fetch(`http://${hostname}:8000/load/${uuid}/rgb_frame/${newFrame-1}`)
            .then((res) => {
                if (!res.ok) {
                    throw new Error(`Could not retrieve frame=${newFrame} for job with uuid=${uuid}`)
                }
                return res.json()
            })
            .catch((error) => {
                setMessage('Could not retrieve requested image data')
                return [[]];
            })
    };

    const loadRunData = async () => {
        return await fetch(`http://${hostname}:8000/load/${uuid}/run_data`)
            .then((res) => {
                if (!res.ok) {
                    throw new Error(`Could not retrieve run data for job with uuid=${uuid}`)
                }
                setValidUuid(true);
                let body = res.json()
                console.log(body)
                return body
            })
            .catch((error) => {
                setMessage('Could not retrieve requested run data')
                setValidUuid(false);
                return {};
            })
    };

    const [exp1, setExp1] = useState<string>('y**2+2*x*y+2*x**2-4-0.5*sin(15*x)');
    const [exp2, setExp2] = useState<string>('y-10*x**2+3');
    const [colourScheme, setColourScheme] = useState<number>(1);
    const [uuid, setUuid] = useState<string>('0887d67e-0970-41b5-b04b-8ff917cd4c7a');
    const [maxFrame, setMaxFrame] = useState<number>(1);
    const [frame, setFrame] = useState<number>(1);
    const [frameCache, setFrameCache] = useState<Cache>({});
    const [validUuid, setValidUuid] = useState<boolean>(false);


    const [message, setMessage] = useState('');

    const hostname = window.location.hostname === 'localhost' ? 'localhost' : '192.168.0.106';

    const handleGenerateClick = async () => {
        generateArray();
    }

    const handleLoadClick = async () => {
        const data = await loadRunData();
        if (data['frames']) {
            setMaxFrame(data['frames']);
        }
    }

    function checkArray(rgbArray: number[][]) {
        if (rgbArray) {
            if (rgbArray.length > 0) {
                if (rgbArray[0].length > 0) {
                    return <ImageRenderer rgbArray={rgbArray} />
                }
            }
        }
        return null;
    }

    useEffect(() => {
        const cachingFrameFetch = async () => {
            if (validUuid) {
                if (frameCache[frame]) {
                    console.debug(`Using cached data for frame: ${frame}`);
                } else {
                    console.debug(`Fetching new data for frame: ${frame}`);
                    const fetchedArray = await loadArray(frame);
                    setFrameCache((prevCache) => ({...prevCache, [frame]: fetchedArray}));
                }
            }
        }

        cachingFrameFetch();
    }, [frame, frameCache, validUuid])

    const range = (start: number, end: number) => Array.from({length: (end - start)}, (v, k) => k + start);

    return (
        <ChakraProvider>
            <div>
                    <h1>Basins of Attraction</h1>
                

                    <h2>New Creation</h2>
                    <Flex alignItems='center' borderWidth='3px' borderRadius='lg'>
                        <VStack align='left'>
                            <Box fontSize='small' borderWidth='3px' borderRadius='lg'>Expressions</Box>
                            <Input type='text' variant='filled' width='500px' value={exp1} onChange={e => setExp1(e.target.value)} title={'Expression 1'}/>
                            <Input type='text' variant='filled' width='500px' value={exp2} onChange={e => setExp2(e.target.value)} title={'Expression 2'}/>
                        </VStack>
                        <Spacer />
                        <Center height='50px'>
                            <Divider orientation='vertical' />
                        </Center>
                        <Spacer />
                        <div>
                            <Box fontSize='small' borderWidth='3px' borderRadius='lg'>Color Scheme</Box>
                            <div><Select name="colourScheme" id="colourScheme" variant='filled' value={colourScheme} onChange={e => setColourScheme(Number(e.target.value))}>
                                {range(1, 11).map((x) => (
                                    <option value={x}>{x}</option>
                                ))}
                            </Select></div>    
                        </div>
                        <Spacer />
                        <Center height='50px'>
                            <Divider orientation='vertical' />
                        </Center>
                        <Spacer />
                        <div>
                            <Button type='submit' colorScheme='green' onClick={handleGenerateClick}>Generate</Button>
                        </div>
                    </Flex>

                    <h2>Load Previous Creation</h2>
                    <Flex alignItems='center' borderWidth='3px' borderRadius='lg'>
                        <Input type='text' variant='filled' value={uuid} onChange={e => {
                            setUuid(e.target.value)
                        }} placeholder='Enter an ID'/>
                        <Button type='submit' colorScheme='green' onClick={handleLoadClick}>Load</Button>
                        <SliderStepInput minValue={1} maxValue={maxFrame} onChange={(v: number) => {
                            setFrame(v);
                        }}/>
                    </Flex>
                
                <Box alignItems='center' borderWidth='3px' borderRadius='lg'>
                    {checkArray(frameCache[frame])}
                </Box>
            </div>
        </ChakraProvider>
    );
};

export default App;
