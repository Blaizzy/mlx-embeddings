# frozen_string_literal: true

RSpec.describe Homebrew, :integration_test do
  it "does not invoke `require \"formula\"` at startup" do
    expect { brew "verify-formula-undefined" }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
