document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = window.location.origin;
    let JWT_TOKEN = null;
    let CURRENT_ADMIN_INFO = { username: '', uid: -1 };

    const AVAILABLE_ROLES = {
        'ROLE_DATA_CONTRIBUTOR': 'Kann Trainingsdaten beitragen',
        'ROLE_MODEL_CONSUMER': 'Kann Modelle herunterladen',
        'ROLE_TOWER_TRAINER': 'Kann Modelle hochladen & Batches verwalten',
        'ROLE_USER_ADMIN': 'Kann Benutzer verwalten'
    };

    const dom = {
        app: document.getElementById('app'),
        loginScreen: document.getElementById('login-screen'),
        dashboard: document.getElementById('dashboard'),
        loginForm: document.getElementById('login-form'),
        loginError: document.getElementById('login-error'),
        loginButton: document.getElementById('login-button'),
        currentAdminUsername: document.getElementById('current-admin-username'),
        userTableBody: document.getElementById('user-table-body'),
        loadingSpinner: document.getElementById('loading-spinner'),
        
        createUserModal: document.getElementById('create-user-modal'),
        createUserModalContent: document.getElementById('create-user-modal-content'),
        showCreateUserModalButton: document.getElementById('show-create-user-modal-button'),
        createUserForm: document.getElementById('create-user-form'),
        newRolesContainer: document.getElementById('new-roles-container'),

        editUserModal: document.getElementById('edit-user-modal'),
        editModalContent: document.getElementById('edit-user-modal-content'),
        editUsernameDisplay: document.getElementById('edit-username-display'),
        editUserId: document.getElementById('edit-user-id'),
        editRolesContainer: document.getElementById('edit-roles-container'),
        saveRolesButton: document.getElementById('save-roles-button'),
        resetPasswordButton: document.getElementById('reset-password-button'),
        resetApikeyButton: document.getElementById('reset-apikey-button'),
        
        showCredentialModal: document.getElementById('show-credential-modal'),
        credentialModalContent: document.getElementById('show-credential-modal-content'),
        credentialModalTitle: document.getElementById('credential-modal-title'),
        newCredentialElement: document.getElementById('new-credential'),
        copyCredentialButton: document.getElementById('copy-credential-button'),
        copyCredentialFeedback: document.getElementById('copy-credential-feedback'),
    };

    // --- API-Funktionen ---
    async function login(username, password) {
        dom.loginButton.disabled = true;
        dom.loginButton.textContent = "Melde an...";
        try {
            const formData = new URLSearchParams({ username, password });
            const response = await fetch(`${API_BASE_URL}/admin/token`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(errorData.detail || `HTTP-Fehler ${response.status}`);
            }
            const data = await response.json();
            JWT_TOKEN = data.access_token;
            const decodedToken = JSON.parse(atob(JWT_TOKEN.split('.')[1]));
            CURRENT_ADMIN_INFO = { username: decodedToken.sub, uid: decodedToken.uid };
            dom.currentAdminUsername.textContent = CURRENT_ADMIN_INFO.username;
            dom.loginScreen.classList.add('hidden');
            dom.dashboard.classList.remove('hidden');
            await fetchUsers();
        } catch (error) {
            dom.loginError.textContent = `Login fehlgeschlagen: ${error.message}`;
            dom.loginError.classList.remove('hidden');
        } finally {
            dom.loginButton.disabled = false;
            dom.loginButton.textContent = "Anmelden";
        }
    }

    async function fetchWithToken(url, options = {}) {
        if (!JWT_TOKEN) throw new Error("Nicht authentifiziert. Bitte neu anmelden.");
        const headers = { ...options.headers, 'Authorization': `Bearer ${JWT_TOKEN}`, 'accept': 'application/json' };
        const response = await fetch(url, { ...options, headers });
        
        if (response.status === 401) { JWT_TOKEN = null; throw new Error("Sitzung abgelaufen oder ungültig."); }
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `HTTP-Fehler ${response.status}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return response.json();
        }
        return true;
    }
    
    async function fetchUsers() {
        dom.loadingSpinner.classList.remove('hidden');
        dom.userTableBody.innerHTML = '';
        try {
            const users = await fetchWithToken(`${API_BASE_URL}/admin/users`);
            renderUserTable(users, CURRENT_ADMIN_INFO.uid);
        } catch (error) {
            handleApiError(error);
        } finally {
            dom.loadingSpinner.classList.add('hidden');
        }
    }
    
    // --- UI-Rendering ---
    function renderUserTable(users, loggedInAdminId) {
        if (!users || users.length === 0) {
            dom.userTableBody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-gray-500">Keine Benutzer gefunden.</td></tr>';
            return;
        }
        dom.userTableBody.innerHTML = users.map(user => {
            const isMainAdmin = user.id === 1;
            const canBeEdited = !(isMainAdmin && loggedInAdminId !== 1);
            const disabledAttribute = canBeEdited ? '' : 'disabled title="Haupt-Admin kann nicht von anderen geändert werden"';
            
            return `
            <tr class="hover:bg-gray-50">
                <td class="px-6 py-4 text-sm font-medium text-gray-900">${user.id}</td>
                <td class="px-6 py-4 text-sm text-gray-700">${user.username}</td>
                <td class="px-6 py-4 text-sm text-gray-500 break-all">${user.roles.split(',').join(', ')}</td>
                <td class="px-6 py-4 text-center">
                    <button class="toggle-status-btn btn btn-sm ${user.is_active ? 'btn-success' : 'btn-danger'}" 
                            data-user-id="${user.id}" data-current-status="${user.is_active}" ${disabledAttribute}>
                        ${user.is_active ? 'Aktiv' : 'Inaktiv'}
                    </button>
                </td>
                <td class="px-6 py-4 text-center">
                    <button class="edit-user-btn btn btn-secondary btn-sm" 
                            data-user-id="${user.id}" data-username="${user.username}" data-roles="${user.roles}" ${disabledAttribute}>
                        Bearbeiten
                    </button>
                </td>
            </tr>`;
        }).join('');
    }

    function renderRolesCheckboxes(container, selectedRoles = []) {
        container.innerHTML = '';
        Object.entries(AVAILABLE_ROLES).forEach(([role, description]) => {
            const isChecked = selectedRoles.includes(role);
            const label = document.createElement('label');
            label.className = 'checkbox-label';
            label.innerHTML = `
                <input type="checkbox" class="checkbox" value="${role}" ${isChecked ? 'checked' : ''}>
                <span class="ml-2 text-sm">
                    <strong>${role}</strong>: 
                    <span class="text-gray-600">${description}</span>
                </span>
            `;
            container.appendChild(label);
        });
    }

    function getSelectedRoles(container) {
        const checkboxes = container.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(cb => cb.value).join(',');
    }


    // --- Modal-Handling & Aktionen ---
   
    function openModal(modal) {
        const content = modal.querySelector('.modal-content');
        modal.classList.remove('hidden');
        setTimeout(() => { modal.style.opacity = '1';
        if(content) content.style.transform = 'scale(1)';
        }, 10);
    }
    function closeModal(modal) {
        const content = modal.querySelector('.modal-content');
        modal.style.opacity = '0';
        if(content) content.style.transform = 'scale(0.95)';
        setTimeout(() => { modal.classList.add('hidden'); }, 250);
    }

    function showNewCredential(title, credential) {
        dom.credentialModalTitle.textContent = title;
        dom.newCredentialElement.textContent = credential;
        dom.copyCredentialFeedback.textContent = '';
        openModal(dom.showCredentialModal);
    }
    
    function handleApiError(error) {
        alert(`Ein Fehler ist aufgetreten: ${error.message}`);
        if (JWT_TOKEN === null) {
            dom.dashboard.classList.add('hidden');
            dom.loginScreen.classList.remove('hidden');
        }
    }

    // --- Event Listeners ---
    dom.loginForm.addEventListener('submit', async e => {
        e.preventDefault();
        dom.loginError.classList.add('hidden');
        const username = document.getElementById('username-input').value.trim();
        const password = document.getElementById('password-input').value.trim();
        if(username && password) {
            await login(username, password);
        }
    });
    
    dom.showCreateUserModalButton.addEventListener('click', () => {
        dom.createUserForm.reset();
        const defaultRoles = ['ROLE_DATA_CONTRIBUTOR', 'ROLE_MODEL_CONSUMER'];
        renderRolesCheckboxes(dom.newRolesContainer, defaultRoles);
        openModal(dom.createUserModal, dom.createUserModalContent);
    });

    dom.createUserForm.addEventListener('submit', e => {
        e.preventDefault();
        const username = document.getElementById('new-username').value.trim();
        const password = document.getElementById('new-password').value.trim();
        const roles = getSelectedRoles(dom.newRolesContainer);
        if (username && password && roles) {
            fetchWithToken(`${API_BASE_URL}/admin/users`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password, roles, notes: "Erstellt via Web-UI" })
            }).then(data => {
                showNewCredential('API-Key für neuen Benutzer', data.api_key);
                closeModal(dom.createUserModal, dom.createUserModalContent);
                fetchUsers();
            }).catch(handleApiError);
        }
    });
    
    dom.userTableBody.addEventListener('click', e => {
        const button = e.target.closest('button');
        if (!button) return;

        const userId = button.dataset.userId;

        if (button.classList.contains('edit-user-btn')) {
            const userData = button.dataset;
            dom.editUserId.value = userData.userId;
            dom.editUsernameDisplay.textContent = userData.username;
            renderRolesCheckboxes(dom.editRolesContainer, userData.roles.split(','));
            openModal(dom.editUserModal);
        } 
        else if (button.classList.contains('toggle-status-btn')) {
            const currentStatus = button.dataset.currentStatus === 'true';
            if (confirm(`Möchten Sie den Status des Benutzers wirklich auf '${!currentStatus ? 'Aktiv' : 'Inaktiv'}' setzen?`)) {
                fetchWithToken(`${API_BASE_URL}/admin/users/${userId}/status`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ is_active: !currentStatus })
                }).then(fetchUsers).catch(handleApiError);
            }
        }
    });
    
    dom.saveRolesButton.addEventListener('click', () => {
        const userId = dom.editUserId.value;
        const newRoles = getSelectedRoles(dom.editRolesContainer);
        if (!newRoles) {
            alert("Mindestens eine Rolle muss ausgewählt sein.");
            return;
        }
        fetchWithToken(`${API_BASE_URL}/admin/users/${userId}/roles`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ roles: newRoles })
        }).then(() => {
            alert('Rollen erfolgreich aktualisiert!');
            closeModal(dom.editUserModal);
            fetchUsers();
        }).catch(handleApiError);
    });

    dom.resetPasswordButton.addEventListener('click', () => {
        const userId = dom.editUserId.value;
        if (confirm('Möchten Sie das Passwort für diesen Benutzer wirklich zurücksetzen?')) {
            fetchWithToken(`${API_BASE_URL}/admin/users/${userId}/reset-password`, { method: 'PUT' })
                .then(data => {
                    closeModal(dom.editUserModal);
                    showNewCredential('Neues Passwort', data.new_password);
                }).catch(handleApiError);
        }
    });
    
    dom.resetApikeyButton.addEventListener('click', () => {
        const userId = dom.editUserId.value;
        if (confirm('Möchten Sie den API-Key für diesen Benutzer wirklich zurücksetzen?')) {
            fetchWithToken(`${API_BASE_URL}/admin/users/${userId}/reset-apikey`, { method: 'PUT' })
                .then(data => {
                    closeModal(dom.editUserModal);
                    showNewCredential('Neuer API-Key', data.new_api_key);
                }).catch(handleApiError);
        }
    });

    document.querySelectorAll('.close-modal-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const modal = btn.closest('.modal');
            if (modal) {
                const content = modal.querySelector('.modal-content');
                closeModal(modal, content);
            }
        });
    });
    
    dom.app.addEventListener('click', e => {
        if(e.target.closest('.close-modal-btn')) {
            const modal = e.target.closest('.modal');
            if (modal) closeModal(modal);
        }
    });

    dom.copyCredentialButton.addEventListener('click', async () => {
        const keyToCopy = dom.newCredentialElement.textContent;
        if (!navigator.clipboard) {
            dom.copyCredentialFeedback.textContent = 'Kopieren nicht unterstützt (kein HTTPS?).';
            return;
        }
        try {
            await navigator.clipboard.writeText(keyToCopy);
            dom.copyCredentialFeedback.textContent = 'Kopiert!';
        } catch (err) {
            dom.copyCredentialFeedback.textContent = 'Fehler beim Kopieren!';
            console.error('Fehler beim Kopieren: ', err);
        } finally {
            setTimeout(() => { dom.copyCredentialFeedback.textContent = '' }, 2000);
        }
    });
});

