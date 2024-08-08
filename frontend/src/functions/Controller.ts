export const fetchPost = async (url: string, request: string): Promise<Response> => {
    return await fetch(url, {
        method: 'POST',
        headers: {
          'Accept': 'application/json, text/plain, */*',
          'Content-Type': 'application/json'
        },
        body: request
        })
}