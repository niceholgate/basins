import React from 'react';

type ImageRendererProps = {
    rgbArray: number[][];
};

const ImageRenderer: React.FC<ImageRendererProps> = ({ rgbArray }) => {
    const generateImageDataUrl = (rgbArray: number[][]) => {
        const canvas = document.createElement('canvas');
        canvas.width = rgbArray[0].length/3;
        canvas.height = rgbArray.length;
        const ctx = canvas.getContext('2d');

        if (ctx) {
            const imageData = ctx.createImageData(canvas.width, canvas.height);

            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const index = (y*canvas.width+x)*4;
                    imageData.data[index] = rgbArray[y][x*3]; // Red
                    imageData.data[index + 1] = rgbArray[y][x*3+1]; // Green
                    imageData.data[index + 2] = rgbArray[y][x*3+2]; // Blue
                    imageData.data[index + 3] = 255; // Alpha
                }
            }

            ctx.putImageData(imageData, 0, 0);
            return canvas.toDataURL();
        }

        return '';
    };

    const imageUrl = generateImageDataUrl(rgbArray);

    return <img src={imageUrl} alt='RGB data rendered' width={500}/>;
};

export default ImageRenderer;