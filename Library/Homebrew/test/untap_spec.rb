# frozen_string_literal: true

require "untap"

RSpec.describe Homebrew::Untap do
  describe ".installed_formulae_for" do
    shared_examples "finds installed formulae in tap" do
      def load_formula(name:, with_formula_file: false, mock_install: false)
        formula = formula(name, tap:) do
          url "https://brew.sh/foo-1.0.tgz"
        end

        if with_formula_file
          class_name = name.split("_").map(&:capitalize).join
          tap.formula_dir.mkpath
          (tap.formula_dir/"#{name}.rb").write <<~RUBY
            class #{class_name} < Formula
              url "https://brew.sh/foo-1.0.tgz"
            end
          RUBY
        end

        if mock_install
          keg_path = HOMEBREW_CELLAR/name/"1.2.3"
          keg_path.mkpath

          tab_path = keg_path/Tab::FILENAME
          tab_path.write <<~JSON
            {
              "source": {
                "tap": "#{tap}"
              }
            }
          JSON
        end

        formula
      end

      let!(:currently_installed_formula) do
        load_formula(name: "current_install", with_formula_file: true, mock_install: true)
      end

      before do
        # Formula that is available from a tap but not installed.
        load_formula(name: "no_install", with_formula_file: true)

        # Formula that was installed from a tap but is no longer available from that tap.
        load_formula(name: "legacy_install", mock_install: true)

        tap.clear_cache
      end

      it "returns the expected formulae" do
        expect(described_class.installed_formulae_for(tap:).map(&:full_name))
          .to eq([currently_installed_formula.full_name])
      end
    end

    context "with core tap" do
      let(:tap) { CoreTap.instance }

      include_examples "finds installed formulae in tap"
    end

    context "with non-core tap" do
      let(:tap) { Tap.fetch("homebrew", "foo") }

      include_examples "finds installed formulae in tap"
    end
  end

  describe ".installed_casks_for" do
    shared_examples "finds installed casks in tap" do
      def load_cask(token:, with_cask_file: false, mock_install: false)
        cask_loader = Cask::CaskLoader::FromContentLoader.new(<<~RUBY, tap:)
          cask '#{token}' do
            version "1.2.3"
            sha256 :no_check

            url 'https://brew.sh/'
          end
        RUBY

        cask = cask_loader.load(config: nil)

        if with_cask_file
          cask_path = tap.cask_dir/"#{token}.rb"
          cask_path.parent.mkpath
          cask_path.write cask.source
        end

        if mock_install
          metadata_subdirectory = cask.metadata_subdir("Casks", timestamp: :now, create: true)
          (metadata_subdirectory/"#{token}.rb").write cask.source
        end

        cask
      end

      let!(:currently_installed_cask) do
        load_cask(token: "current_install", with_cask_file: true, mock_install: true)
      end

      before do
        # Cask that is available from a tap but not installed.
        load_cask(token: "no_install", with_cask_file: true)

        # Cask that was installed from a tap but is no longer available from that tap.
        load_cask(token: "legacy_install", mock_install: true)
      end

      it "returns the expected casks" do
        expect(described_class.installed_casks_for(tap:)).to eq([currently_installed_cask])
      end
    end

    context "with core cask tap" do
      let(:tap) { CoreCaskTap.instance }

      include_examples "finds installed casks in tap"
    end

    context "with non-core cask tap" do
      let(:tap) { Tap.fetch("homebrew", "foo") }

      include_examples "finds installed casks in tap"
    end
  end
end
