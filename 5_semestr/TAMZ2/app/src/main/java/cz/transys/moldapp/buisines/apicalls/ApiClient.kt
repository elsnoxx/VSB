package cz.transys.moldapp.buisines.apicalls

import io.ktor.client.*
import io.ktor.client.engine.okhttp.*
import io.ktor.client.call.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.serialization.gson.*

object ApiClient {

    const val BASE_URL = "https://mesapi.hyundai-transys.cz/api/"

    val client = HttpClient(OkHttp) {
        install(ContentNegotiation) {
            gson()
        }
    }

    // universal GET
    suspend inline fun <reified T> get(path: String): T {
        return client.get(BASE_URL + path).body()
    }

    // universal POST
    suspend inline fun <reified T> post(path: String, payload: Any): T {
        return client.post(BASE_URL + path) {
            setBody(payload)
        }.body()
    }
}
