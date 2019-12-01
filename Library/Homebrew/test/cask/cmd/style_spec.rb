# frozen_string_literal: true

require "open3"

require_relative "shared_examples/invalid_option"

describe Cask::Cmd::Style, :cask do
  let(:args) { [] }
  let(:cli) { described_class.new(*args) }

  it_behaves_like "a command that handles invalid options"

  describe ".rubocop" do
    subject { described_class.rubocop(cask_path) }

    around do |example|
      FileUtils.ln_s HOMEBREW_LIBRARY_PATH, HOMEBREW_LIBRARY/"Homebrew"
      FileUtils.ln_s HOMEBREW_LIBRARY_PATH.parent/".rubocop_cask.yml", HOMEBREW_LIBRARY/".rubocop_cask.yml"
      FileUtils.ln_s HOMEBREW_LIBRARY_PATH.parent/".rubocop_shared.yml", HOMEBREW_LIBRARY/".rubocop_shared.yml"

      example.run
    ensure
      FileUtils.rm_f HOMEBREW_LIBRARY/"Homebrew"
      FileUtils.rm_f HOMEBREW_LIBRARY/".rubocop_cask.yml"
      FileUtils.rm_f HOMEBREW_LIBRARY/".rubocop_shared.yml"
    end

    before do
      allow(Homebrew).to receive(:install_bundler_gems!)
    end

    context "with a valid Cask" do
      let(:cask_path) do
        Pathname.new("#{HOMEBREW_LIBRARY}/Homebrew/test/support/fixtures/cask/Casks/version-latest.rb")
      end

      it "returns true" do
        expect(subject.success?).to be true
      end
    end
  end

  describe "#cask_paths" do
    subject { cli.cask_paths }

    before do
      allow(cli).to receive(:args).and_return(tokens)
    end

    context "when no cask tokens are given" do
      let(:tokens) { [] }

      matcher :a_path_ending_with do |end_string|
        match do |actual|
          expect(actual.to_s).to end_with(end_string)
        end
      end

      it {
        expect(subject).to contain_exactly(
          a_path_ending_with("/homebrew/homebrew-cask/Casks"),
          a_path_ending_with("/third-party/homebrew-tap/Casks"),
          a_path_ending_with("/Homebrew/test/support/fixtures/cask/Casks"),
          a_path_ending_with("/Homebrew/test/support/fixtures/third-party/Casks"),
        )
      }
    end

    context "when at least one cask token is a path that exists" do
      let(:tokens) { ["adium", "Casks/dropbox.rb"] }

      before do
        allow(File).to receive(:exist?).and_return(false, true)
      end

      it "treats all tokens as paths" do
        expect(subject).to eq [
          Pathname("adium").expand_path,
          Pathname("Casks/dropbox.rb").expand_path,
        ]
      end
    end

    context "when no cask tokens are paths that exist" do
      let(:tokens) { %w[adium dropbox] }

      before do
        allow(File).to receive(:exist?).and_return(false)
      end

      it "tries to find paths for all tokens" do
        expect(Cask::CaskLoader).to receive(:load).twice.and_return(double("cask", sourcefile_path: nil))
        subject
      end
    end
  end
end
