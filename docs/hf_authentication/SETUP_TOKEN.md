# 🔐 Hugging Face Token Setup Guide

## Step 1: Generate Your Hugging Face Token

1. **Go to Hugging Face Settings**: https://huggingface.co/settings/tokens
2. **Click "New token"**
3. **Settings**:
   - **Name**: `Dataset Upload Token` (or any descriptive name)
   - **Type**: **Write** ⚠️ (Must be Write, not Read!)
   - **Expiration**: 90 days (recommended for security)
4. **Click "Generate a token"**
5. **Copy the token immediately** (you won't see it again!)

## Step 2: Create Your .env File

1. **Copy the template**:
   ```cmd
   copy .env.template .env
   ```

2. **Edit the .env file** and replace `your_write_token_here` with your actual token:
   ```
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

3. **Save the file**

## Step 3: Install Requirements

```cmd
pip install -r requirements_upload.txt
```

## Step 4: Run the Upload Script

```cmd
python src/upload_to_huggingface.py
```

The script will automatically:
- ✅ Load your token from the .env file
- ✅ Authenticate with Hugging Face
- ✅ Upload your selected dataset

## 🛡️ Security Notes

- **Never commit .env files** to version control (already added to .gitignore)
- **Use Write tokens only when needed**
- **Set expiration dates** on your tokens
- **Revoke old/unused tokens** from your HF settings

## 🔄 Alternative Token Names

The script looks for tokens in this order:
1. `HUGGINGFACE_TOKEN`
2. `HF_TOKEN`

You can use either variable name in your .env file.

## 🆘 Troubleshooting

### "Authentication failed"
- Check that your token is correct in .env file
- Ensure token has **Write** permissions
- Verify token hasn't expired

### "No module named 'dotenv'"
```cmd
pip install python-dotenv
```

### "Token not found"
- Make sure .env file is in the same directory as the script
- Check that the variable name is exactly `HUGGINGFACE_TOKEN`
- Ensure no extra spaces around the = sign

## 📁 File Structure

```
AI_MasterBlackBelt/
├── .env                    # Your token file (not in git)
├── .env.template          # Template for reference
├── .gitignore            # Keeps .env secure
├── src/
│   └── upload_to_huggingface.py
├── requirements_upload.txt
└── datasets/
    ├── lss_consultant/
    └── lss_ner/
```
