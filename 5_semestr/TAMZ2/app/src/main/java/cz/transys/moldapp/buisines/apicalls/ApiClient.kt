package cz.transys.moldapp.buisines.apicalls

import android.util.Log
import cz.transys.moldapp.buisines.models.TokenStore
import io.ktor.client.*
import io.ktor.client.engine.okhttp.*
import io.ktor.client.call.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.serialization.gson.*
import io.ktor.http.*
import io.ktor.client.plugins.auth.Auth
import io.ktor.client.plugins.auth.providers.BearerTokens
import io.ktor.client.plugins.auth.providers.bearer

object ApiClient {

    const val BASE_URL = "https://mesapi.hyundai-transys.cz/api/"

    val client = HttpClient(OkHttp) {
        install(ContentNegotiation) {
            gson()
        }
    }

    val authClient = HttpClient(OkHttp) {
        install(ContentNegotiation) {
            gson()
        }

        install(Auth) {
            bearer {
                loadTokens {
                    val jwt = TokenStore.jwt
                    if (jwt.isNullOrBlank()) null
                    else BearerTokens(accessToken = jwt, refreshToken = "")
                }
            }
        }
    }

    // Public GET for login
    suspend inline fun <reified T> getPublic(path: String): T {
        Log.d("API call to", "$BASE_URL + $path")
        return client.get(BASE_URL + path).body()
    }

    // universal GET
    suspend inline fun <reified T> get(path: String): T {
        Log.d("API call to", "$BASE_URL + $path")
        return authClient.get(BASE_URL + path).body()
    }

    // universal POST
    suspend inline fun <reified T> post(path: String, payload: Any): T {
        Log.d("API call to", "$BASE_URL + $path")
        return authClient.post(BASE_URL + path) {
            contentType(ContentType.Application.Json)
            setBody(payload)
        }.body()
    }
}
