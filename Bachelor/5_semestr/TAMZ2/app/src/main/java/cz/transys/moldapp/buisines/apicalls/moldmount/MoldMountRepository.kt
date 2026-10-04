package cz.transys.moldapp.buisines.apicalls.moldmount

import cz.transys.moldapp.buisines.apicalls.ApiClient
import cz.transys.moldapp.buisines.apicalls.ApiResponse

class MoldMountRepository {
    suspend fun postMoldMount(moldMount: MoldMountResquest): ApiResponse {
        return ApiClient.post("foampad/moldpda/mount", moldMount)
    }
}

