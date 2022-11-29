const BASE_URL = 'http://localhost:1113'

const APP = [
    'models',
    'upload',
    'detect'
]

const app = (() => {
    let obj = {}
    APP.forEach((value) => {
        obj[value] = `${BASE_URL}/app/${value}`
    })
    return obj
})()

const exported = {
    BASE_URL,
    app
}

export default exported;