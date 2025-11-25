package cz.transys.moldapp.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.R
import cz.transys.moldapp.ui.apicalls.CarriersList
import cz.transys.moldapp.ui.apicalls.MoldApiRepository

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MoldMountScreen(onBack: () -> Unit) {
    val colors = MaterialTheme.colorScheme

    val repo = remember { MoldApiRepository() }
    val scope = rememberCoroutineScope()
    // Carrier ComboBox
    var expandedCarrier by remember { mutableStateOf(false) }
    var selectedCarrier by remember { mutableStateOf("") }
    var carrierList by remember { mutableStateOf<List<CarriersList>>(emptyList()) }

    // Mold #1
    var mold1Type by remember { mutableStateOf("rrr") }
    var mold1Code by remember { mutableStateOf("rrr") }

    // Mold #2
    var mold2Type by remember { mutableStateOf("rrr") }
    var mold2Code by remember { mutableStateOf("rrr") }

    LaunchedEffect(Unit) {
        try {
            carrierList = repo.getAllCarriers()
        } catch (e: Exception) {
            // TODO případně ukázat error
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        Text(
            text = stringResource(R.string.mold_mount_title),
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = colors.primary,
            modifier = Modifier.padding(bottom = 10.dp)
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.background)
                .padding(8.dp)
        ) {

            ExposedDropdownMenuBox(
                expanded = expandedCarrier,
                onExpandedChange = { expandedCarrier = !expandedCarrier }
            ) {

                OutlinedTextField(
                    value = selectedCarrier,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text(stringResource(R.string.carrier_label)) },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedCarrier)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                )

                ExposedDropdownMenu(
                    expanded = expandedCarrier,
                    onDismissRequest = { expandedCarrier = false }
                ) {

                    carrierList.forEach { carrier ->

                        DropdownMenuItem(
                            text = { Text(carrier.code_value2) }, // "#01"
                            onClick = {
                                selectedCarrier = carrier.code_value1 // "C01"
                                expandedCarrier = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Column(
                modifier = Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
            ) {

                // BOX A – current molds
                MoldBox(
                    title = "Current Molds",
                    headerColor = colors.tertiary,
                    mold1 = {
                        MoldSectionCompact(
                            title = stringResource(R.string.mold1_title),
                            color = colors.tertiary,
                            carCode = selectedCarrier,
                            type = mold1Type,
                            code = mold1Code,
                            readOnlyCode = true,
                            onCodeChange = {}
                        )
                    },
                    mold2 = {
                        MoldSectionCompact(
                            title = stringResource(R.string.mold2_title),
                            color = colors.tertiary,
                            carCode = selectedCarrier,
                            type = mold2Type,
                            code = mold2Code,
                            readOnlyCode = true,
                            onCodeChange = {}
                        )
                    }
                )

                Spacer(modifier = Modifier.height(16.dp))

                // BOX B – new molds to mount
                MoldBox(
                    title = "New Molds",
                    headerColor = colors.error,
                    mold1 = {
                        MoldSectionCompact(
                            title = stringResource(R.string.mold1_title),
                            color = colors.error,
                            carCode = selectedCarrier,
                            type = mold1Type,
                            code = mold1Code,
                            readOnlyCode = false,
                            onCodeChange = { mold1Code = it }
                        )
                    },
                    mold2 = {
                        MoldSectionCompact(
                            title = stringResource(R.string.mold2_title),
                            color = colors.error,
                            carCode = selectedCarrier,
                            type = mold2Type,
                            code = mold2Code,
                            readOnlyCode = false,
                            onCodeChange = { mold2Code = it }
                        )
                    }
                )
            }



            Spacer(modifier = Modifier.height(20.dp))

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // SUBMIT / MOUNT – větší tlačítko
                Button(
                    onClick = {
                        // TODO: API call
                    },
                    modifier = Modifier
                        .weight(0.7f)
                        .height(40.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colors.primary,
                        contentColor = colors.onPrimary
                    )
                ) {
                    Text(
                        text = stringResource(R.string.mount_button),
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )
                }

                // BACK – menší tlačítko
                Button(
                    onClick = onBack,
                    modifier = Modifier
                        .weight(0.3f)
                        .height(40.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colors.secondary,
                        contentColor = colors.onSecondary
                    )
                ) {
                    Text(
                        text = stringResource(R.string.back_button),
                        fontSize = 16.sp
                    )
                }
            }

        }
    }
}

@Composable
fun MoldBox(
    title: String,
    headerColor: Color,
    mold1: @Composable () -> Unit,
    mold2: @Composable () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(headerColor.copy(alpha = 0.10f), MaterialTheme.shapes.small)
            .padding(3.dp)
    ) {
        Text(
            text = title,
            color = headerColor,
            fontWeight = FontWeight.Bold,
            fontSize = 8.sp,
            modifier = Modifier.padding(bottom = 4.dp)
        )

        mold1()

        Spacer(modifier = Modifier.height(3.dp))

        mold2()
    }
}



@Composable
fun ReadOnlyFieldMini(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Text(label, fontSize = 8.sp, color = Color.Gray)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White, MaterialTheme.shapes.small)
                .padding(4.dp)
        ) {
            Text(value.ifBlank { "-" })
        }
    }
}


@Composable
fun MoldSectionCompact(
    title: String,
    color: Color,
    carCode: String,
    type: String,
    code: String,
    readOnlyCode: Boolean = false,
    onCodeChange: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(color.copy(alpha = 0.10f), MaterialTheme.shapes.small)
            .padding(2.dp)
    ) {

        Text(
            text = title,
            fontSize = 9.sp,
            fontWeight = FontWeight.Bold,
            color = color,
            modifier = Modifier.padding(bottom = 2.dp)
        )

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {

            // 1. řádek – Car + Type
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                MiniReadOnlyBox(
                    label = "Car",
                    value = carCode,
                    modifier = Modifier.weight(0.25f)
                )

                MiniReadOnlyBox(
                    label = "Type",
                    value = type,
                    modifier = Modifier.weight(0.75f)
                )
            }

            // 2. řádek – Mold code (full width)
            MiniReadOnlyBox(
                label = "Code",
                value = code,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

@Composable
fun MiniReadOnlyBox(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {

        Text(
            text = label,
            fontSize = 7.sp,          // ↓ menší
            color = Color.Gray,
            modifier = Modifier.padding(bottom = 1.dp)
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White, MaterialTheme.shapes.extraSmall)
                .padding(horizontal = 4.dp, vertical = 2.dp) // ↓ menší padding
        ) {
            Text(
                text = value.ifBlank { "-" },
                fontSize = 9.sp        // ↓ menší text
            )
        }
    }
}







