import React, { useState } from "react";
import {
    Flex,
    NumberInput,
    NumberInputField,
    NumberInputStepper,
    NumberIncrementStepper,
    NumberDecrementStepper,
    Slider,
    SliderTrack,
    SliderFilledTrack,
    SliderThumb } from "@chakra-ui/react";

type SliderStepInputProps = {
    frame: number;
    setFrame: Function;
    minValue: number;
    maxValue: number;
    onChange: Function;
};

const SliderStepInput: React.FC<SliderStepInputProps> = ({ frame, setFrame, minValue, maxValue, onChange }) => {
    //const [value, setValue] = useState(minValue)
    const handleChange = (newVal: number) => {
      setFrame(newVal)
      onChange(newVal)
    }
    const handleChangeString = (newVal: string) => {
      setFrame(parseInt(newVal))
      onChange(parseInt(newVal))
    }
  
    return (
      <Flex>
        <NumberInput maxW='100px' mr='2rem' value={frame} min={minValue} max={maxValue} onChange={handleChangeString}>
          <NumberInputField />
          <NumberInputStepper>
            <NumberIncrementStepper />
            <NumberDecrementStepper />
          </NumberInputStepper>
        </NumberInput>
        <Slider
          flex='1'
          focusThumbOnChange={false}
          value={frame}
          min={minValue}
          max={maxValue}
          onChange={handleChange}
        >
          <SliderTrack>
            <SliderFilledTrack />
          </SliderTrack>
          <SliderThumb fontSize='sm' boxSize='32px' children={frame} />
        </Slider>
      </Flex>
    )
  }

  export default SliderStepInput;