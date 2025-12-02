package cz.transys.moldapp.buisines.apicalls.moldmount

import cz.transys.moldapp.buisines.apicalls.ApiClient

class MoldMountRepository {
    suspend fun postMoldMount(moldMount: MoldMountResquest): Boolean {
        return ApiClient.post("foampad/moldpda/mount", moldMount)
    }
}

