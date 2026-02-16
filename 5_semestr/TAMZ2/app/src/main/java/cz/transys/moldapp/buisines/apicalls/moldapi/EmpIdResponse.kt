package cz.transys.moldapp.buisines.apicalls.moldapi

data class EmpIdResponse(
    val emp_name: String,
    val message: String,
    val emp_id: String,
    val token: String,
    val result: String
)