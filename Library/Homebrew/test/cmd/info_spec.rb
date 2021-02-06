# typed: false
# frozen_string_literal: true

require "cmd/info"

require "cmd/shared_examples/args_parse"

describe "brew info" do
  let(:tarball) do
    if OS.linux?
      TEST_FIXTURE_DIR/"tarballs/testball-0.1-linux.tbz"
    else
      TEST_FIXTURE_DIR/"tarballs/testball-0.1.tbz"
    end
  end
  let(:expected_output) {
    <<~EOS
      {"formulae":[{"name":"testball","full_name":"testball","tap":"homebrew/core","oldname":null,"aliases":[],"versioned_formulae":[],"desc":"Some test","license":null,"homepage":"https://brew.sh/testball","versions":{"stable":"0.1","head":null,"bottle":false},"urls":{"stable":{"url":"file://#{tarball}","tag":null,"revision":null}},"revision":0,"version_scheme":0,"bottle":{},"keg_only":false,"bottle_disabled":false,"options":[{"option":"--with-foo","description":"Build with foo"}],"build_dependencies":[],"dependencies":[],"recommended_dependencies":[],"optional_dependencies":[],"uses_from_macos":[],"requirements":[],"conflicts_with":[],"caveats":null,"installed":[],"linked_keg":null,"pinned":false,"outdated":false,"deprecated":false,"deprecation_date":null,"deprecation_reason":null,"disabled":false,"disable_date":null,"disable_reason":null}],"casks":[]}
    EOS
  }

  it_behaves_like "parseable arguments"

  it "prints as json with the --json=v1 flag", :integration_test do
    setup_test_formula "testball"

    expect { brew "info", "testball", "--json=v1" }
      .to output(a_json_string).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "prints as json with the --json=v2 flag", :integration_test do
    setup_test_formula "testball"

    expect { brew "info", "testball", "--json=v2" }
      .to output(a_json_string).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "check --json=v2 format", :integration_test do
    setup_test_formula "testball"

    expect { brew "info", "testball", "--json=v2" }
      .to output(expected_output).to_stdout
  end

  describe Homebrew do
    describe "::github_remote_path" do
      let(:remote) { "https://github.com/Homebrew/homebrew-core" }

      specify "returns correct URLs" do
        expect(described_class.github_remote_path(remote, "Formula/git.rb"))
          .to eq("https://github.com/Homebrew/homebrew-core/blob/HEAD/Formula/git.rb")

        expect(described_class.github_remote_path("#{remote}.git", "Formula/git.rb"))
          .to eq("https://github.com/Homebrew/homebrew-core/blob/HEAD/Formula/git.rb")

        expect(described_class.github_remote_path("git@github.com:user/repo", "foo.rb"))
          .to eq("https://github.com/user/repo/blob/HEAD/foo.rb")

        expect(described_class.github_remote_path("https://mywebsite.com", "foo/bar.rb"))
          .to eq("https://mywebsite.com/foo/bar.rb")
      end
    end
  end
end
