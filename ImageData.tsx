// import React, { useEffect, useState } from "react";
// import {
//     Box,
//     Button,
//     Flex,
//     Input,
//     InputGroup,
//     Modal,
//     ModalBody,
//     ModalCloseButton,
//     ModalContent,
//     ModalFooter,
//     ModalHeader,
//     ModalOverlay,
//     Stack,
//     Text,
//     useDisclosure
// } from "@chakra-ui/react";
//
// const ImageDataContext = React.createContext({
//   imageData: [], fetchImageData: () => {}
// })
//
// export default function ImageData() {
//     const [imageData, setImageData] = useState([])
//     const fetchImageData = async () => {
//         const response = await fetch("http://localhost:8000/load/rgb_frame/6d1ee19b-2549-452d-98e1-2948b7354f92/0")
//         const imageData = await response.json()
//         setImageData(imageData.data)
//     }
// }
//
// useEffect(() => {
//   fetchImageData()
// }, [])
//
// return (
//   <TodosContext.Provider value={{imageData, fetchImageData}}>
//     <Stack spacing={5}>
//       {todos.map((todo) => (
//         <b>{todo.item}</b>
//       ))}
//     </Stack>
//   </TodosContext.Provider>
// )