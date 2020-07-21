# frozen_string_literal: true

describe Cask::Audit, :cask do
  def include_msg?(messages, msg)
    if msg.is_a?(Regexp)
      Array(messages).any? { |m| m =~ msg }
    else
      Array(messages).include?(msg)
    end
  end

  matcher :pass do
    match do |audit|
      !audit.errors? && !audit.warnings?
    end
  end

  matcher :fail_with do |error_msg|
    match do |audit|
      include_msg?(audit.errors, error_msg)
    end
  end

  matcher :warn_with do |warning_msg|
    match do |audit|
      include_msg?(audit.warnings, warning_msg)
    end
  end

  let(:cask) { instance_double(Cask::Cask) }
  let(:download) { false }
  let(:token_conflicts) { false }
  let(:strict) { false }
  let(:fake_system_command) { class_double(SystemCommand) }
  let(:audit) {
    described_class.new(cask, download:        download,
                              token_conflicts: token_conflicts,
                              command:         fake_system_command,
                              strict:          strict)
  }

  describe "#result" do
    subject { audit.result }

    context "when there are errors" do
      before do
        audit.add_error "bad"
      end

      it { is_expected.to match(/failed/) }
    end

    context "when there are warnings" do
      before do
        audit.add_warning "eh"
      end

      it { is_expected.to match(/warning/) }
    end

    context "when there are errors and warnings" do
      before do
        audit.add_error "bad"
        audit.add_warning "eh"
      end

      it { is_expected.to match(/failed/) }
    end

    context "when there are no errors or warnings" do
      it { is_expected.to match(/passed/) }
    end
  end

  describe "#run!" do
    subject { audit.run! }

    def tmp_cask(name, text)
      path = Pathname.new "#{dir}/#{name}.rb"
      path.open("w") do |f|
        f.write text
      end

      Cask::CaskLoader.load(path)
    end

    let(:dir) { mktmpdir }
    let(:cask) { Cask::CaskLoader.load(cask_token) }

    describe "required stanzas" do
      %w[version sha256 url name homepage].each do |stanza|
        context "when missing #{stanza}" do
          let(:cask_token) { "missing-#{stanza}" }

          it { is_expected.to fail_with(/#{stanza} stanza is required/) }
        end
      end
    end

    describe "token validation" do
      let(:strict) { true }
      let(:cask) do
        tmp_cask cask_token.to_s, <<~RUBY
          cask '#{cask_token}' do
            version '1.0'
            sha256 '8dd95daa037ac02455435446ec7bc737b34567afe9156af7d20b2a83805c1d8a'
            url "https://brew.sh/"
            name 'Audit'
            homepage 'https://brew.sh/'
            app 'Audit.app'
          end
        RUBY
      end

      context "when cask token is not lowercase" do
        let(:cask_token) { "Upper-Case" }

        it "warns about lowercase" do
          expect(subject).to warn_with(/token is not lowercase/)
        end
      end

      context "when cask token is not ascii" do
        let(:cask_token) { "ascii⌘" }

        it "warns about ascii" do
          expect(subject).to warn_with(/contains non-ascii characters/)
        end
      end

      context "when cask token has +" do
        let(:cask_token) { "app++" }

        it "warns about +" do
          expect(subject).to warn_with(/\+ should be replaced by -plus-/)
        end
      end

      context "when cask token has @" do
        let(:cask_token) { "app@stuff" }

        it "warns about +" do
          expect(subject).to warn_with(/@ should be replaced by -at-/)
        end
      end

      context "when cask token has whitespace" do
        let(:cask_token) { "app stuff" }

        it "warns about whitespace" do
          expect(subject).to warn_with(/whitespace should be replaced by hyphens/)
        end
      end

      context "when cask token has underscores" do
        let(:cask_token) { "app_stuff" }

        it "warns about underscores" do
          expect(subject).to warn_with(/underscores should be replaced by hyphens/)
        end
      end

      context "when cask token has non-alphanumeric characters" do
        let(:cask_token) { "app(stuff)" }

        it "warns about non-alphanumeric characters" do
          expect(subject).to warn_with(/should only contain alphanumeric characters and hyphens/)
        end
      end

      context "when cask token has double hyphens" do
        let(:cask_token) { "app--stuff" }

        it "warns about double hyphens" do
          expect(subject).to warn_with(/should not contain double hyphens/)
        end
      end

      context "when cask token has trailing hyphens" do
        let(:cask_token) { "app-" }

        it "warns about trailing hyphens" do
          expect(subject).to warn_with(/should not have leading or trailing hyphens/)
        end
      end
    end

    describe "token bad words" do
      let(:strict) { true }
      let(:cask) do
        tmp_cask cask_token.to_s, <<~RUBY
          cask '#{cask_token}' do
            version '1.0'
            sha256 '8dd95daa037ac02455435446ec7bc737b34567afe9156af7d20b2a83805c1d8a'
            url "https://brew.sh/"
            name 'Audit'
            homepage 'https://brew.sh/'
            app 'Audit.app'
          end
        RUBY
      end

      context "when cask token contains .app" do
        let(:cask_token) { "token.app" }

        it "warns about .app" do
          expect(subject).to warn_with(/token contains .app/)
        end
      end

      context "when cask token contains version" do
        let(:cask_token) { "token-beta" }

        it "warns about version in token" do
          expect(subject).to warn_with(/token contains version/)
        end
      end

      context "when cask token contains launcher" do
        let(:cask_token) { "token-launcher" }

        it "warns about launcher in token" do
          expect(subject).to warn_with(/token mentions launcher/)
        end
      end

      context "when cask token contains desktop" do
        let(:cask_token) { "token-desktop" }

        it "warns about desktop in token" do
          expect(subject).to warn_with(/token mentions desktop/)
        end
      end

      context "when cask token contains platform" do
        let(:cask_token) { "token-osx" }

        it "warns about platform in token" do
          expect(subject).to warn_with(/token mentions platform/)
        end
      end

      context "when cask token contains architecture" do
        let(:cask_token) { "token-x86" }

        it "warns about architecture in token" do
          expect(subject).to warn_with(/token mentions architecture/)
        end
      end

      context "when cask token contains framework" do
        let(:cask_token) { "token-java" }

        it "warns about framework in token" do
          expect(subject).to warn_with(/cask token mentions framework/)
        end
      end

      context "when cask token is framework" do
        let(:cask_token) { "java" }

        it "does not warn about framework" do
          expect(subject).not_to warn_with(/token contains version/)
        end
      end
    end

    describe "locale validation" do
      let(:strict) { true }
      let(:cask) do
        tmp_cask "locale-cask-test", <<~RUBY
          cask 'locale-cask-test' do
            version '1.0'
            url "https://brew.sh/"
            name 'Audit'
            homepage 'https://brew.sh/'
            app 'Audit.app'

            language 'en', default: true do
              sha256 '96574251b885c12b48a3495e843e434f9174e02bb83121b578e17d9dbebf1ffb'
              'zh-CN'
            end

            language 'zh-CN' do
              sha256 '96574251b885c12b48a3495e843e434f9174e02bb83121b578e17d9dbebf1ffb'
              'zh-CN'
            end

            language 'ZH-CN' do
              sha256 '96574251b885c12b48a3495e843e434f9174e02bb83121b578e17d9dbebf1ffb'
              'zh-CN'
            end

            language 'zh-' do
              sha256 '96574251b885c12b48a3495e843e434f9174e02bb83121b578e17d9dbebf1ffb'
              'zh-CN'
            end

            language 'zh-cn' do
              sha256 '96574251b885c12b48a3495e843e434f9174e02bb83121b578e17d9dbebf1ffb'
              'zh-CN'
            end
          end
        RUBY
      end

      context "when cask locale is invalid" do
        it "error with invalid locale" do
          expect(subject).to fail_with(/locale ZH-CN, zh-, zh-cn are invalid/)
        end
      end
    end

    describe "pkg allow_untrusted checks" do
      let(:warning_msg) { "allow_untrusted is not permitted in official Homebrew Cask taps" }

      context "when the Cask has no pkg stanza" do
        let(:cask_token) { "basic-cask" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask does not have allow_untrusted" do
        let(:cask_token) { "with-uninstall-pkgutil" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has allow_untrusted" do
        let(:cask_token) { "with-allow-untrusted" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "when the Cask stanza requires uninstall" do
      let(:warning_msg) { "installer and pkg stanzas require an uninstall stanza" }

      context "when the Cask does not require an uninstall" do
        let(:cask_token) { "basic-cask" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the pkg Cask has an uninstall" do
        let(:cask_token) { "with-uninstall-pkgutil" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the installer Cask has an uninstall" do
        let(:cask_token) { "installer-with-uninstall" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the installer Cask does not have an uninstall" do
        let(:cask_token) { "with-installer-manual" }

        it { is_expected.to warn_with(warning_msg) }
      end

      context "when the pkg Cask does not have an uninstall" do
        let(:cask_token) { "pkg-without-uninstall" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "preflight stanza checks" do
      let(:warning_msg) { "only a single preflight stanza is allowed" }

      context "when the Cask has no preflight stanza" do
        let(:cask_token) { "with-zap-rmdir" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has only one preflight stanza" do
        let(:cask_token) { "with-preflight" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has multiple preflight stanzas" do
        let(:cask_token) { "with-preflight-multi" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "uninstall_postflight stanza checks" do
      let(:warning_msg) { "only a single postflight stanza is allowed" }

      context "when the Cask has no postflight stanza" do
        let(:cask_token) { "with-zap-rmdir" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has only one postflight stanza" do
        let(:cask_token) { "with-postflight" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has multiple postflight stanzas" do
        let(:cask_token) { "with-postflight-multi" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "uninstall stanza checks" do
      let(:warning_msg) { "only a single uninstall stanza is allowed" }

      context "when the Cask has no uninstall stanza" do
        let(:cask_token) { "with-zap-rmdir" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has only one uninstall stanza" do
        let(:cask_token) { "with-uninstall-rmdir" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has multiple uninstall stanzas" do
        let(:cask_token) { "with-uninstall-multi" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "uninstall_preflight stanza checks" do
      let(:warning_msg) { "only a single uninstall_preflight stanza is allowed" }

      context "when the Cask has no uninstall_preflight stanza" do
        let(:cask_token) { "with-zap-rmdir" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has only one uninstall_preflight stanza" do
        let(:cask_token) { "with-uninstall-preflight" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has multiple uninstall_preflight stanzas" do
        let(:cask_token) { "with-uninstall-preflight-multi" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "uninstall_postflight stanza checks" do
      let(:warning_msg) { "only a single uninstall_postflight stanza is allowed" }

      context "when the Cask has no uninstall_postflight stanza" do
        let(:cask_token) { "with-zap-rmdir" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has only one uninstall_postflight stanza" do
        let(:cask_token) { "with-uninstall-postflight" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has multiple uninstall_postflight stanzas" do
        let(:cask_token) { "with-uninstall-postflight-multi" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "zap stanza checks" do
      let(:warning_msg) { "only a single zap stanza is allowed" }

      context "when the Cask has no zap stanza" do
        let(:cask_token) { "with-uninstall-rmdir" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has only one zap stanza" do
        let(:cask_token) { "with-zap-rmdir" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask has multiple zap stanzas" do
        let(:cask_token) { "with-zap-multi" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "version checks" do
      let(:error_msg) { "you should use version :latest instead of version 'latest'" }

      context "when version is 'latest'" do
        let(:cask_token) { "version-latest-string" }

        it { is_expected.to fail_with(error_msg) }
      end

      context "when version is :latest" do
        let(:cask_token) { "version-latest-with-checksum" }

        it { is_expected.not_to fail_with(error_msg) }
      end
    end

    describe "sha256 checks" do
      context "when version is :latest and sha256 is not :no_check" do
        let(:cask_token) { "version-latest-with-checksum" }

        it { is_expected.to fail_with("you should use sha256 :no_check when version is :latest") }
      end

      context "when sha256 is not a legal SHA-256 digest" do
        let(:cask_token) { "invalid-sha256" }

        it { is_expected.to fail_with("sha256 string must be of 64 hexadecimal characters") }
      end

      context "when sha256 is sha256 for empty string" do
        let(:cask_token) { "sha256-for-empty-string" }

        it { is_expected.to fail_with(/cannot use the sha256 for an empty string/) }
      end
    end

    describe "hosting with appcast checks" do
      let(:appcast_warning) { /please add an appcast/ }

      context "when the download does not use hosting with an appcast" do
        let(:cask_token) { "basic-cask" }

        it { is_expected.not_to warn_with(appcast_warning) }
      end

      context "when the download uses GitHub releases and has an appcast" do
        let(:cask_token) { "github-with-appcast" }

        it { is_expected.not_to warn_with(appcast_warning) }
      end

      context "when the download uses GitHub releases and does not have an appcast" do
        let(:cask_token) { "github-without-appcast" }

        it { is_expected.to warn_with(appcast_warning) }
      end

      context "when the download is hosted on SourceForge and has an appcast" do
        let(:cask_token) { "sourceforge-with-appcast" }

        it { is_expected.not_to warn_with(appcast_warning) }
      end

      context "when the download is hosted on SourceForge and does not have an appcast" do
        let(:cask_token) { "sourceforge-correct-url-format" }

        it { is_expected.to warn_with(appcast_warning) }
      end

      context "when the download is hosted on DevMate and has an appcast" do
        let(:cask_token) { "devmate-with-appcast" }

        it { is_expected.not_to warn_with(appcast_warning) }
      end

      context "when the download is hosted on DevMate and does not have an appcast" do
        let(:cask_token) { "devmate-without-appcast" }

        it { is_expected.to warn_with(appcast_warning) }
      end

      context "when the download is hosted on HockeyApp and has an appcast" do
        let(:cask_token) { "hockeyapp-with-appcast" }

        it { is_expected.not_to warn_with(appcast_warning) }
      end

      context "when the download is hosted on HockeyApp and does not have an appcast" do
        let(:cask_token) { "hockeyapp-without-appcast" }

        it { is_expected.to warn_with(appcast_warning) }
      end
    end

    describe "latest with appcast checks" do
      let(:warning_msg) { "Casks with an appcast should not use version :latest" }

      context "when the Cask is :latest and does not have an appcast" do
        let(:cask_token) { "version-latest" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask is versioned and has an appcast" do
        let(:cask_token) { "with-appcast" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask is :latest and has an appcast" do
        let(:cask_token) { "latest-with-appcast" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "denylist checks" do
      context "when the Cask isn't disallowed" do
        let(:cask_token) { "adobe-air" }

        it { is_expected.to pass }
      end

      context "when the Cask is disallowed" do
        context "and it's in the official Homebrew tap" do
          let(:cask_token) { "adobe-illustrator" }

          it { is_expected.to fail_with(/#{cask_token} is not allowed: \w+/) }
        end

        context "and it isn't in the official Homebrew tap" do
          let(:cask_token) { "pharo" }

          it { is_expected.to pass }
        end
      end
    end

    describe "latest with auto_updates checks" do
      let(:warning_msg) { "Casks with `version :latest` should not use `auto_updates`" }

      context "when the Cask is :latest and does not have auto_updates" do
        let(:cask_token) { "version-latest" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask is versioned and does not have auto_updates" do
        let(:cask_token) { "basic-cask" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask is versioned and has auto_updates" do
        let(:cask_token) { "auto-updates" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "when the Cask is :latest and has auto_updates" do
        let(:cask_token) { "latest-with-auto-updates" }

        it { is_expected.to warn_with(warning_msg) }
      end
    end

    describe "preferred download URL formats" do
      let(:warning_msg) { /URL format incorrect/ }

      context "with incorrect SourceForge URL format" do
        let(:cask_token) { "sourceforge-incorrect-url-format" }

        it { is_expected.to warn_with(warning_msg) }
      end

      context "with correct SourceForge URL format" do
        let(:cask_token) { "sourceforge-correct-url-format" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "with correct SourceForge URL format for version :latest" do
        let(:cask_token) { "sourceforge-version-latest-correct-url-format" }

        it { is_expected.not_to warn_with(warning_msg) }
      end

      context "with incorrect OSDN URL format" do
        let(:cask_token) { "osdn-incorrect-url-format" }

        it { is_expected.to warn_with(warning_msg) }
      end

      context "with correct OSDN URL format" do
        let(:cask_token) { "osdn-correct-url-format" }

        it { is_expected.not_to warn_with(warning_msg) }
      end
    end

    describe "generic artifact checks" do
      context "with relative target" do
        let(:cask_token) { "generic-artifact-relative-target" }

        it { is_expected.to fail_with(/target must be absolute path for Generic Artifact/) }
      end

      context "with absolute target" do
        let(:cask_token) { "generic-artifact-absolute-target" }

        it { is_expected.not_to fail_with(/target required for Generic Artifact/) }
      end
    end

    describe "url checks" do
      context "given a block" do
        let(:cask_token) { "booby-trap" }

        context "when loading the cask" do
          it "does not evaluate the block" do
            expect { cask }.not_to raise_error
          end
        end

        context "when doing the audit" do
          it "evaluates the block" do
            expect(subject).to fail_with(/Boom/)
          end
        end
      end
    end

    describe "token conflicts" do
      let(:cask_token) { "with-binary" }
      let(:token_conflicts) { true }

      context "when cask token conflicts with a core formula" do
        let(:formula_names) { %w[with-binary other-formula] }

        it "warns about duplicates" do
          expect(audit).to receive(:core_formula_names).and_return(formula_names)
          expect(subject).to warn_with(/possible duplicate/)
        end
      end

      context "when cask token does not conflict with a core formula" do
        let(:formula_names) { %w[other-formula] }

        it { is_expected.not_to warn_with(/possible duplicate/) }
      end
    end

    describe "audit of downloads" do
      let(:cask_token) { "with-binary" }
      let(:cask) { Cask::CaskLoader.load(cask_token) }
      let(:download_double) { instance_double(Cask::Download) }
      let(:verify) { class_double(Cask::Verify).as_stubbed_const }
      let(:error_msg) { "Download Failed" }

      before do
        allow(audit).to receive(:download).and_return(download_double)
      end

      it "when download and verification succeed it does not fail" do
        expect(download_double).to receive(:perform)
        expect(verify).to receive(:all)
        expect(subject).not_to fail_with(/#{error_msg}/)
      end

      it "when download fails it does not fail" do
        expect(download_double).to receive(:perform).and_raise(StandardError.new(error_msg))
        expect(subject).to fail_with(/#{error_msg}/)
      end

      it "when verification fails it does not fail" do
        expect(download_double).to receive(:perform)
        expect(verify).to receive(:all).and_raise(StandardError.new(error_msg))
        expect(subject).to fail_with(/#{error_msg}/)
      end
    end

    context "when an exception is raised" do
      let(:cask) { instance_double(Cask::Cask) }

      it "fails the audit" do
        expect(cask).to receive(:tap).and_raise(StandardError.new)
        expect(subject).to fail_with(/exception while auditing/)
      end
    end
  end
end
