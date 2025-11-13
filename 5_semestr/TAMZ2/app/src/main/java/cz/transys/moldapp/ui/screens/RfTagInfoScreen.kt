package cz.transys.moldapp.ui.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.LocalScanner
import cz.transys.moldapp.ui.apicalls.MoldRfInfoRepository
import cz.transys.moldapp.ui.apicalls.TagInfo
import kotlinx.coroutines.launch

@Composable
fun RfTagInfoScreen(onBack: () -> Unit) {
    val repo = remember { MoldRfInfoRepository() }
    val scope = rememberCoroutineScope()
    var rfTag by remember { mutableStateOf("") }

    // placeholder data (později se naplní z API)
    val fields = listOf(
        "Mold ID",
        "Part Type",
        "Last Used Date",
        "Operator",
        "Location"
    )

    var info by remember { mutableStateOf<TagInfo?>(null) }
    var loading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    fun loadData(tag: String) {
        scope.launch {
            loading = true
            error = null
            info = null
            try {
                Log.d("RFINFO", "Calling API: getTagInfo($tag)")
                val result = repo.getTagInfo(tag)
                if (result == null) {
                    Log.w("RFINFO", "API returned NULL (tag not found)")
                    error = "RF-TAG nebyl nalezen v systém"
                } else {
                    Log.d("RFINFO", "Parsed API DATA:")
                    info = result
                }
            } catch (e: Exception) {
                Log.e("RFINFO", "ERROR in loadData(): ${e.localizedMessage}", e)
                error = "API chyba: ${e.localizedMessage}"
            } finally {
                loading = false
                Log.d("RFINFO", "loadData() END (loading=$loading, error=$error, info=$info)")
                Log.d("RFINFO", "==============================")
            }
        }
    }

    fun onScanned(value: String) {
        val clean = value.trim()
        rfTag = clean
        if (clean.isNotEmpty()) {
            loadData(clean)   // 🔥 po naskenování rovnou voláme API
        }
    }

    // Honeywell scanner listener
    val scanner = LocalScanner.current
    LaunchedEffect(scanner) {
        scanner?.setOnScanListener { scannedData ->
            onScanned(scannedData)
        }
    }
    DisposableEffect(scanner) {
        onDispose { scanner?.setOnScanListener { } }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFECECEC))
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Nadpis
        Text(
            text = "RF-Tag Information",
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF1565C0),
            modifier = Modifier.padding(bottom = 20.dp)
        )

        // Formulářové okno (pozadí bílé, obdélníky)
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(16.dp)
        ) {
            // RF-TAG vstupní pole
            OutlinedTextField(
                value = rfTag,
                onValueChange = { rfTag = it },
                label = { Text("RF-TAG") },
                placeholder = { Text("Načti nebo zadej kód") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Placeholder hodnoty (později API)
            fields.forEach { label ->
                if (info != null) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFFF2F2F2), shape = MaterialTheme.shapes.medium)
                            .padding(12.dp)
                            .padding(top = 8.dp, bottom = 8.dp)
                    ) {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {

                            ReadOnlyField(label = "Mold ID", value = info!!.mold)
                            ReadOnlyField(label = "Part Type", value = info!!.type)
                            ReadOnlyField(label = "Car", value = info!!.car)
                            ReadOnlyField(label = "Status", value = info!!.status)
                            ReadOnlyField(label = "Total Cycles", value = info!!.total)
                        }
                    }
                }
            }




            Spacer(modifier = Modifier.height(20.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                // 🔵 LOAD DATA
                Button(
                    onClick = { /* TODO: API call */ },
                    modifier = Modifier
                        .weight(1f)
                        .height(60.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF1565C0),
                        contentColor = Color.White
                    )
                ) {
                    Text(
                        text = "LOAD DATA",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )
                }

                // ⚪ BACK
                Button(
                    onClick = onBack,
                    modifier = Modifier
                        .weight(.5f)
                        .height(60.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.Gray,
                        contentColor = Color.White
                    )
                ) {
                    Text(
                        text = "Zpět",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )
                }
            }
        }
    }


}

@Composable
fun ReadOnlyField(label: String, value: String) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = Color.Gray
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White, MaterialTheme.shapes.small)
                .padding(10.dp)
        ) {
            Text(
                text = value.ifBlank { "-" },
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}