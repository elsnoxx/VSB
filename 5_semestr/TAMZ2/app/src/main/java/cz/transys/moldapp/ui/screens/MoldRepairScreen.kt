package cz.transys.moldapp.ui.screens

import android.annotation.SuppressLint
import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.LocalScanner
import cz.transys.moldapp.ui.apicalls.ApiClient
import cz.transys.moldapp.ui.apicalls.MoldRepairRepository
import cz.transys.moldapp.ui.apicalls.RepairTypes
import cz.transys.moldapp.ui.localdata.LocalStorage
import kotlinx.coroutines.launch
import android.provider.Settings
import android.widget.Toast
import androidx.compose.runtime.setValue
import androidx.compose.ui.res.stringResource
import cz.transys.moldapp.R
import cz.transys.moldapp.ui.apicalls.CarrierMountResponse
import cz.transys.moldapp.ui.apicalls.CarriersList
import cz.transys.moldapp.ui.apicalls.MoldApiRepository
import cz.transys.moldapp.ui.apicalls.MoldRepairSent

@SuppressLint("HardwareIds")
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MoldRepairScreen(onBack: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    // api
    val repairRepo = remember { MoldRepairRepository() }
    val repo = remember { MoldApiRepository() }
    val scope = rememberCoroutineScope()
    // Stav načítání
    var error by remember { mutableStateOf<String?>(null) }
    var loading by remember { mutableStateOf(true) }
    val scanner = LocalScanner.current

    var carrierList by remember { mutableStateOf<List<CarriersList>>(emptyList()) }
    var mountInfo by remember { mutableStateOf<CarrierMountResponse?>(null) }

    // variables
    val context = LocalContext.current
    val deviceId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
    val storage = remember { LocalStorage(context) }
    var mold by remember { mutableStateOf("") }
    var outDate by remember { mutableStateOf("") }
    var carType by remember { mutableStateOf("") }
    var type by remember { mutableStateOf("") }

    // comboBox – TYPE
    var expanded by remember { mutableStateOf(false) }
    var selectedType by remember { mutableStateOf("") }

    // zatím testovací statická data
    var typeList by remember { mutableStateOf<List<RepairTypes>>(emptyList()) }

    fun validateAndProceed(data: String) {
        val clean = data.trim()
        mold = clean

        Log.d("MOLD_SCAN", "Scanned: '$data' → Cleaned: '$clean'")

        if (clean.isNotEmpty()) {
            scope.launch {
                try {
                    loading = true

                    Log.d("MOLD_API", "Calling getMoldRepairInfo('$clean')")

                    val info = repairRepo.getMoldRepairInfo(clean)

                    Log.d("MOLD_API", "API Result: $info")

                    if (info == null) {
                        Log.w("MOLD_API", "Mold not found for code: $clean")

                        error = context.getString(R.string.mold_not_found)
                        carType = ""
                        outDate = ""
                        type = ""

                    } else {

                        Log.d("MOLD_DATA", """
                        ✔ Mapping API response:
                        car_code    = ${info.car_code}
                        save_dttm   = ${info.save_dttm}
                        mold_name   = ${info.mold_name}
                    """.trimIndent())

                        carType = info.car_code
                        outDate = info.save_dttm
                        type = info.mold_name

                        error = null
                    }

                } catch (e: Exception) {

                    Log.e(
                        "MOLD_API_ERROR",
                        "Exception while calling API: ${e.localizedMessage}",
                        e
                    )

                    error = context.getString(R.string.api_error_full, e.localizedMessage)

                } finally {
                    loading = false
                    Log.d("MOLD_STATE", "Loading finished. loading=$loading")
                }
            }
        }
    }


    // Zaregistrujeme listener na sken
    LaunchedEffect(scanner) {
        scanner?.setOnScanListener { scannedData ->
            validateAndProceed(scannedData)
        }

        loading = true
        try {
            typeList = repairRepo.getAllRepairTypes()
            Log.d("Repair Types API", "Repair types: $typeList")
        } catch (e: Exception) {
            error = e.localizedMessage
        } finally {
            loading = false
        }
    }

    DisposableEffect(scanner) {
        onDispose {
            // po opuštění obrazovky zrušíme listener
            scanner?.setOnScanListener { }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        // Title
        Text(
            text = stringResource(id = R.string.mold_repair_title),
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = colors.primary,
            modifier = Modifier.padding(bottom = 20.dp)
        )

        // From for mold and type
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.surface, MaterialTheme.shapes.medium)
                .padding(16.dp)
        ) {
            // Mold
            OutlinedTextField(
                value = mold,
                onValueChange = {},
                readOnly = true,
                label = { Text(stringResource(id = R.string.mold_label)) },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = colors.primary,
                    unfocusedBorderColor = colors.outline,
                    focusedLabelColor = colors.primary,
                    cursorColor = colors.primary
                )
            )


            // Type (ComboBox)
            ExposedDropdownMenuBox(
                expanded = expanded,
                onExpandedChange = { expanded = !expanded }
            ) {
                OutlinedTextField(
                    value = selectedType,
                    onValueChange = { },
                    readOnly = true,
                    label = { Text(stringResource(id = R.string.type_label)) },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                    },
                    modifier = Modifier
                        .menuAnchor()
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = colors.primary,
                        unfocusedBorderColor = colors.outline,
                        focusedLabelColor = colors.primary
                    )
                )

                ExposedDropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    typeList.forEach { type ->
                        DropdownMenuItem(
                            text = { Text(type.repair_name) },
                            onClick = {
                                selectedType = type.repair_name
                                expanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(colors.surfaceVariant, MaterialTheme.shapes.medium)
                    .padding(12.dp)
            ) {

                Column {

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {

                        // left side – Car Code
                        OutlinedTextField(
                            value = carType,
                            onValueChange = { },
                            readOnly = true,
                            label = { Text(stringResource(id = R.string.car_label)) },
                            modifier = Modifier
                                .weight(.5f)
                                .padding(vertical = 4.dp)
                        )

                        // right side – Type
                        OutlinedTextField(
                            value = type,
                            onValueChange = { },
                            readOnly = true,
                            label = { Text(stringResource(id = R.string.type_label)) },
                            modifier = Modifier
                                .weight(1f)
                                .padding(vertical = 4.dp)
                        )
                    }

                    // Out Date
                    OutlinedTextField(
                        value = outDate,
                        onValueChange = { },
                        readOnly = true,
                        label = { Text(stringResource(id = R.string.out_date_label)) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp)
                    )

                    // Repairer
                    OutlinedTextField(
                        value = storage.getUserId().toString(),
                        onValueChange = { },
                        readOnly = true,
                        label = { Text(stringResource(id = R.string.repairer_label)) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp)
                    )
                }
            }


            Spacer(modifier = Modifier.height(20.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                // 🔧 REPAIR button
                Button(
                    onClick = {
                        scope.launch {
                            try {
                                val deviceId = Settings.Secure.getString(
                                    context.contentResolver,
                                    Settings.Secure.ANDROID_ID
                                )

                                if (mold.isBlank() || selectedType.isBlank()) {
                                    Toast.makeText(
                                        context,
                                        context.getString(R.string.error_missing_fields),
                                        Toast.LENGTH_LONG
                                    ).show()
                                    return@launch
                                }

                                val model = MoldRepairSent(
                                    sysId = deviceId,
                                    moldCode = mold,
                                    repairCode = selectedType,
                                    empId = storage.getUserId().toString()
                                )

                                val success = repairRepo.postMoldRepair(model)

                                if (success) {
                                    Toast.makeText(
                                        context,
                                        context.getString(R.string.success_sent),
                                        Toast.LENGTH_SHORT
                                    ).show()
                                }
                            } catch (e: Exception) {
                                Toast.makeText(
                                    context,
                                    context.getString(R.string.api_error_full, e.localizedMessage),
                                    Toast.LENGTH_LONG
                                ).show()
                            }
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(60.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colors.tertiary,
                        contentColor = colors.onTertiary
                    )
                ) {
                    Text(
                        text = stringResource(id = R.string.repair_button),
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp
                    )
                }

                // ZPĚT
                Button(
                    onClick = onBack,
                    modifier = Modifier
                        .weight(.5f)
                        .height(60.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colors.secondary,
                        contentColor = colors.onSecondary
                    )
                ) {
                    Text(stringResource(id = R.string.back_button) )
                }
            }

        }
    }
}