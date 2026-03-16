package main

import (
	"os"

	"github.com/tensorhero/tinytorch-tester/internal/stages"
	tester_utils "github.com/tensorhero/tester-utils"
)

func main() {
	definition := stages.GetDefinition()
	os.Exit(tester_utils.Run(os.Args[1:], definition))
}
