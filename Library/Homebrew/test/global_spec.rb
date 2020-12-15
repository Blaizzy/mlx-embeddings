# typed: false
# frozen_string_literal: true

describe "brew", :integration_test do
  it "does not invoke `require \"formula\"` at startup" do
    expect { brew "verify-formula-undefined" }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "ensures that HOMEBREW_LIBRARY=HOMEBREW_REPOSITORY/Library" do
    expect(HOMEBREW_LIBRARY.to_s).to eq("#{HOMEBREW_REPOSITORY}/Library")
  end
end
