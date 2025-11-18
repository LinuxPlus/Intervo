require('dotenv').config();

// Node.js v18+ has native fetch
const MAILCOACH_TOKEN = process.env.MAILCOACH_TOKEN;
const MAILCOACH_TENANT_URL = process.env.MAILCOACH_TENANT_URL;

// آیا سرویس ایمیل به‌طور کامل تنظیم شده؟
const isEmailServiceConfigured = Boolean(MAILCOACH_TOKEN && MAILCOACH_TENANT_URL);
const isProduction = process.env.NODE_ENV === 'production';

/**
 * Sends a transactional email using Mailcoach.
 *
 * @param {object} emailData - The data for the email.
 * @param {string} emailData.mail_name - The name of the transactional email as specified on the email's screen.
 * @param {string} emailData.subject - The subject of the email.
 * @param {string} emailData.from - The from address to use.
 * @param {string} emailData.to - The to address to use. Multiple addresses can be comma-delimited.
 * @param {string} [emailData.cc] - The cc address to use. Multiple addresses can be comma-delimited.
 * @param {string} [emailData.bcc] - The bcc address to use. Multiple addresses can be comma-delimited.
 * @param {boolean} [emailData.store=true] - Whether to store the email in the transactional email log.
 * @param {string} [emailData.mailer] - The name of the mailer to use.
 * @param {object} [emailData.replacements] - Key-value pairs to replace placeholders in the email template.
 * @param {boolean} [emailData.fake=false] - Whether to actually send the email or just log it.
 * @param {string} [emailData.html] - The HTML body of the email.
 *
 * @returns {Promise<object>} - The response from the Mailcoach API.
 */
async function sendTransactionalEmail(emailData) {
  // --- حالت DEV یا وقتی Mailcoach کانفیگ نشده: فقط لاگ کن، خطا نده ---
  if (!isEmailServiceConfigured || (!isProduction && !emailData.fake)) {
    console.warn('[emailService] Mailcoach is not fully configured OR not in production.');
    console.warn('[emailService] Skipping real email send. Payload:');
    console.log(JSON.stringify(emailData, null, 2));

    // اینجا می‌تونیم لینک جادویی و بقیه اطلاعات رو از لاگ برداریم
    return {
      success: true,
      skipped: true,
      message: 'Email sending skipped (dev mode or Mailcoach not configured)',
    };
  }

  const baseUrl = MAILCOACH_TENANT_URL.replace(/\/$/, '');
  const url = `${baseUrl}/api/transactional-mails/send`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${MAILCOACH_TOKEN}`,
        Accept: "application/json"
      },
      body: JSON.stringify(emailData)
    });

    if (!response.ok) {
      const errorDetail = await response.text();
      throw new Error(
        `Failed to send email: ${response.status} ${response.statusText} - ${errorDetail}`
      );
    }

    console.log('Email sent successfully');
    return response.text();
  } catch (err) {
    console.error('[emailService] Error while calling Mailcoach:', err);
    throw err;
  }
}

/**
 * Sends a magic link email to the user.
 *
 * @param {string} email - The recipient's email address
 * @param {string} magicLink - The complete magic link URL
 * @param {string} firstName - The user's first name (optional)
 * @param {string} type - The type of magic link ('login' or 'signup')
 * @returns {Promise<object>} - The response from the email service
 */
async function sendMagicLinkEmail(email, magicLink, firstName = '', type = 'login') {
  const subject = type === 'login' 
    ? 'Login to your Intervo account' 
    : 'Complete your Intervo account registration';

  const welcomeText = firstName 
    ? `Hi ${firstName},` 
    : 'Hi there,';

  const actionText = type === 'login'
    ? 'to log in to your account'
    : 'to complete your registration';

  try {
    const res = await sendTransactionalEmail({
      mail_name: 'Intervo Verification Email',
      subject: subject,
      from: 'noreply@intervo.ai',
      to: email,
      replacements: {
        welcomeText: welcomeText,
        actionText: actionText,
        magicLink: magicLink,
        userName: firstName || email.split('@')[0],
        expiryTime: '30 minutes'
      }
    });

    return res;
  } catch (error) {
    console.error('Error sending magic link email:', error);
    throw error;
  }
}

/**
 * Sends a workspace invitation email.
 *
 * @param {string} email - The recipient's email address.
 * @param {string} workspaceName - The name of the workspace.
 * @param {string} inviterName - The name of the person who invited the user.
 * @param {string} inviteLink - The URL to accept the invitation.
 * @param {string} [userName] - The recipient's name (if available).
 * @returns {Promise<object>} - The response from the email service.
 */
async function sendWorkspaceInvitationEmail(
  email,
  workspaceName,
  inviterName,
  inviteLink,
  userName = ''
) {
  const subject = `You're invited to join the ${workspaceName} workspace on Intervo`;
  const welcomeText = userName ? `Hi ${userName},` : 'Hi there,';

  try {
    return await sendTransactionalEmail({
      mail_name: 'Intervo Invitaiton Email to Join a Workspace', // Assumed mail_name, please verify
      subject: subject,
      from: 'noreply@intervo.ai',
      to: email,
      replacements: {
        welcomeText: welcomeText,
        userName: userName || email.split('@')[0],
        workspaceName: workspaceName,
        inviterName: inviterName,
        inviteLink: inviteLink,
      },
    });
  } catch (error) {
    console.error('Error sending workspace invitation email:', error);
    return null; // Indicate email sending failed but don't block
  }
}

/**
 * Sends a notification email when a user's access to a workspace is revoked.
 *
 * @param {string} email - The recipient's email address.
 * @param {string} workspaceName - The name of the workspace from which access was revoked.
 * @param {string} [userName] - The recipient's name (if available).
 * @returns {Promise<object|null>} - The response from the email service, or null if sending failed.
 */
async function sendWorkspaceRevocationEmail(email, workspaceName, userName = '') {
  const subject = `Access Revoked: You have been removed from the ${workspaceName} workspace`;
  const welcomeText = userName ? `Hi ${userName},` : 'Hi there,';
  const bodyText = `This email confirms that your access to the workspace "${workspaceName}" on Intervo has been revoked by the workspace owner or an admin. If you believe this was in error, please contact the workspace administrator.`;

  try {
    return await sendTransactionalEmail({
      mail_name: 'Intervo Notification of Workspace Revocation', // As specified
      subject: subject,
      from: 'noreply@intervo.ai',
      to: email,
      replacements: {
        welcomeText: welcomeText,
        userName: userName || email.split('@')[0],
        workspaceName: workspaceName,
        bodyText: bodyText,
      },
    });
  } catch (error) {
    console.error(`Error sending workspace revocation email to ${email}:`, error);
    return null; // Indicate email sending failed but don't block
  }
}

module.exports = {
  sendTransactionalEmail,
  sendMagicLinkEmail,
  sendWorkspaceInvitationEmail,
  sendWorkspaceRevocationEmail
};
