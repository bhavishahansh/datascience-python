import {useState} from "react"

import API from "../services/api"

function Login(){

const[email,setEmail]=useState("")

const[password,setPassword]=useState("")

const handleLogin=async()=>{

try{

const res=await API.post("/login",new URLSearchParams({
                        email:email,

                        password:password

                        })

                    )           

localStorage.setItem("token",res.data.access_token)

alert("Login success")

}

catch{

alert("Login failed")

}

}

return(

<div>

    <h2>Login</h2>

    <input placeholder="email" onChange={(e)=>setEmail(e.target.value)} />

    <input type="password" placeholder="password" onChange={(e)=>setPassword(e.target.value)} />

    <button onClick={handleLogin}>

            Login

    </button>

</div>

)

}

export default Login