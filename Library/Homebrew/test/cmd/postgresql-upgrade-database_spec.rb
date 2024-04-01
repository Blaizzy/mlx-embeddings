# frozen_string_literal: true

require "cmd/postgresql-upgrade-database"
require "cmd/shared_examples/args_parse"

RSpec.describe Homebrew::Cmd::PostgresqlUpgradeDatabase do
  it_behaves_like "parseable arguments"
end
