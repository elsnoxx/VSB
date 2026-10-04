package cz.transys.moldapp.buisines.apicalls.partchange

import android.util.Log
import cz.transys.moldapp.buisines.apicalls.ApiClient
import cz.transys.moldapp.buisines.apicalls.ApiResponse

class PartChangeRepository {
    suspend fun postPartChange(partChange: PartChangeRequest): ApiResponse {
        return ApiClient.post("foampad/moldpda/partchange", partChange)
    }
}