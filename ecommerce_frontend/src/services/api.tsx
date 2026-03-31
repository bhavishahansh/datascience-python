import axios from "axios";

const API = axios.create({
    baseURL: "http://127.0.0.1:8000/api/v1"
})

API.interceptors.request.use((req:any)=>{

    const token=localStorage.getItem("token")

    if(token){

        req.headers.Authorization=`Bearer ${token}`

    }

    return req

})

export default API